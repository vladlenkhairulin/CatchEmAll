#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

def read_table(path, required, optional=()):
    if not path.is_file():
        raise ValueError(f'Не найден файл: {path}')
    with path.open('rb') as stream:
        encoding = 'utf-16' if stream.read(2) in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
    with path.open(encoding=encoding, newline='') as stream:
        first = stream.readline()
        skip = int(first.strip().lower().startswith('sep='))
        header = stream.readline() if skip else first
    def norm(s):
        return re.sub(r'\s+', '_', s.strip().lstrip('\ufeff').lower())
    choices = []
    for sep in (';', ',', '\t', '|'):
        cols = next(csv.reader([header], delimiter=sep))
        names = [norm(c) for c in cols]
        choices.append((len(set(required) & set(names)), sep, cols, names))
    _, sep, cols, names = max(choices, key=lambda x: x[0])
    if len(names) != len(set(names)):
        raise ValueError(f'{path.name}: повторяющиеся названия столбцов')
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f'{path.name}: нет столбцов {sorted(missing)}; прочитаны {names}')
    selected = set(required) | set(optional)
    frame = pd.read_csv(path, sep=sep, encoding=encoding, skiprows=skip,
                        dtype='string', keep_default_na=False,
                        usecols=[c for c, n in zip(cols, names) if n in selected])
    frame.columns = [norm(c) for c in frame.columns]
    for col in frame:
        frame[col] = frame[col].str.strip()
    return frame


def date_values(values):
    return pd.to_datetime(values, format='%Y-%m-%d', errors='coerce')


def time_values(values):
    parts = values.str.extract(r'^(\d{1,2}):(\d{2})(?::(\d{2}))?$')
    h = pd.to_numeric(parts[0], errors='coerce')
    m = pd.to_numeric(parts[1], errors='coerce')
    s = pd.to_numeric(parts[2].fillna('0'), errors='coerce')
    seconds = (h * 3600 + m * 60 + s).where(h.between(0,23) & m.between(0,59) & s.between(0,59))
    return pd.to_timedelta(seconds, unit='s')


def prepare(folder):
    u = read_table(folder/'users.csv', ['user_id'], ['full_name','first_name','last_name','middle_name'])
    f = read_table(folder/'flights.csv', ['flight_id','flight_date','departure_airport','arrival_airport'],
        ['flight_time','arrival_date','arrival_time','flight_number','departure_city','arrival_city'])
    x = read_table(folder/'user_flights.csv', ['user_id','flight_id'])
    for frame, key in [(u,'user_id'),(f,'flight_id')]:
        if frame[key].eq('').any() or frame[key].duplicated().any():
            raise ValueError(f'{key}: пустые или повторяющиеся ключи')
    if not x.user_id.isin(u.user_id).all() or not x.flight_id.isin(f.flight_id).all():
        raise ValueError('user_flights: ссылки на отсутствующих пассажиров или рейсы')
    stats = {'input_links':len(x),'duplicate_links':int(x.duplicated().sum())}
    e = x.drop_duplicates().merge(f,on='flight_id',validate='many_to_one')
    for col in ['flight_time','arrival_date','arrival_time','flight_number','departure_city','arrival_city']:
        if col not in e:
            e[col] = ''
    for col in ['departure_airport','arrival_airport']:
        e[col] = e[col].str.upper()
    equivalent = [c for c in e if c!='flight_id']
    stats['identical_flight_alias_events'] = int(e.duplicated(equivalent).sum())
    e = e.drop_duplicates(equivalent).copy()
    e['dep_day'] = date_values(e.flight_date)
    e['arr_day'] = date_values(e.arrival_date)
    e['dep_clock'] = time_values(e.flight_time)
    e['arr_clock'] = time_values(e.arrival_time)
    e['dep_dt'] = e.dep_day + e.dep_clock
    e['arr_dt'] = e.arr_day + e.arr_clock
    unknown_users = set(e.loc[e.dep_day.isna(),'user_id'])
    stats['users_excluded_unknown_departure_date'] = len(unknown_users)
    excluded = e.loc[e.user_id.isin(unknown_users)].copy()
    e = e.loc[~e.user_id.isin(unknown_users)].copy()
    groups = e.groupby(['user_id','dep_day'],sort=False)
    day_count = groups.flight_id.transform('size')
    known_times = groups.dep_dt.transform('count')
    repeated_times = e.duplicated(['user_id','dep_dt'],keep=False) & e.dep_dt.notna()
    ambiguous = ((day_count>1) & (known_times<day_count)) | repeated_times
    e['ambiguous_order'] = ambiguous
    e['sort_dt'] = e.dep_dt.fillna(e.dep_day)
    e = e.sort_values(['user_id','sort_dt','flight_id'],kind='stable')
    nxt = e.groupby('user_id',sort=False).shift(-1).add_prefix('next_')
    d = pd.concat([e,nxt],axis=1)
    d = d.loc[d.next_flight_id.notna()].copy()
    stats['adjacent_known_flight_pairs'] = len(d)
    d['different_airports'] = d.arrival_airport.ne(d.next_departure_airport)
    valid = d.arrival_airport.str.fullmatch('[A-Z]{3}').fillna(False) & d.next_departure_airport.str.fullmatch('[A-Z]{3}').fillna(False)
    stats['pairs_excluded_missing_airport'] = int((~valid).sum())
    d = d.loc[valid & (d.different_airports | d.ambiguous_order | d.next_ambiguous_order.eq(True))].copy()
    d['hours'] = (d.next_dep_dt-d.arr_dt).dt.total_seconds()/3600
    d['calendar_days'] = (d.next_dep_day-d.arr_day).dt.days
    d['departure_calendar_gap'] = (d.next_dep_day-d.dep_day).dt.days
    d['pair_ambiguous_order'] = d.ambiguous_order | d.next_ambiguous_order.eq(True)
    names = pd.Series('',index=u.index,dtype='string')
    for col in ['first_name','middle_name','last_name']:
        if col in u:
            names = names.str.cat(u[col],sep=' ').str.strip()
    if 'full_name' in u:
        names = names.mask(u.full_name.ne(''),u.full_name)
    u['name'] = names.mask(names.eq(''),u.user_id)
    d = d.merge(u[['user_id','name']],on='user_id',how='left',validate='many_to_one')
    return d, excluded, stats


EXPORT = ['user_id','name','flight_id','flight_number','departure_airport','arrival_airport',
    'arrival_city','flight_date','flight_time','arrival_date','arrival_time',
    'next_flight_id','next_flight_number','next_departure_airport','next_departure_city',
    'next_arrival_airport','next_flight_date','next_flight_time','hours','calendar_days',
    'departure_calendar_gap','reason']


def classify(d, max_days):
    d = d.copy()
    precise = d.arr_dt.notna() & d.next_dep_dt.notna()
    d['reason'] = 'outside_threshold'
    d.loc[~precise & d.calendar_days.between(0,max_days-1),'reason'] = 'calendar_dates_only'
    d.loc[precise & d.hours.ge(0) & d.hours.lt(max_days*24),'reason'] = 'recorded_local_times'
    d.loc[~precise & d.calendar_days.eq(max_days),'reason'] = 'threshold_uncertain_missing_time'
    d.loc[d.arr_day.isna(),'reason'] = 'missing_arrival_date'
    d.loc[(precise & d.hours.lt(0)) | (~precise & d.calendar_days.lt(0)), 'reason'] = 'negative_interval_or_timezone'
    d.loc[d.pair_ambiguous_order,'reason'] = 'ambiguous_flight_order'
    accepted = d.loc[d.reason.isin(['recorded_local_times','calendar_dates_only'])].copy()
    review = d.loc[~d.reason.isin(['recorded_local_times','calendar_dates_only','outside_threshold'])].copy()
    return accepted,review,d


def save(frame,path):
    frame[[c for c in EXPORT if c in frame]].to_csv(path,sep=';',encoding='utf-8-sig',index=False)


def create_html(data,total_users,stats,args,target):
    counts = data.groupby('user_id',sort=False).size().sort_values(ascending=False,kind='stable')
    chosen = counts.head(args.top).index
    items = []
    for uid in chosen:
        g = data.loc[data.user_id.eq(uid)].sort_values('next_sort_dt')
        records = []
        for r in g.itertuples():
            precise = r.reason=='recorded_local_times'
            records.append(dict(airport_a=r.arrival_airport,airport_b=r.next_departure_airport,
                city_a=r.arrival_city,city_b=r.next_departure_city,
                arrival=r.arrival_date+(' '+r.arrival_time if r.arrival_time else ''),
                departure=r.next_flight_date+(' '+r.next_flight_time if r.next_flight_time else ''),
                x_a=r.arr_dt.isoformat() if precise else r.arr_day.isoformat(),
                x_b=r.next_dep_dt.isoformat() if precise else r.next_dep_day.isoformat(),
                flight_a=r.flight_number or r.flight_id,flight_b=r.next_flight_number or r.next_flight_id,
                label=f'{r.hours:.1f} ч' if precise else f'{int(r.calendar_days)} календ. дн.',
                precise=precise))
        items.append(dict(user_id=uid,name=g.name.iloc[0],events=records))
    payload=json.dumps(items,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Смена аэропорта</title>
<style>body{margin:0;background:#f3f5f8;font:15px Arial,sans-serif;color:#172840}
main{max-width:1400px;margin:24px auto;padding:24px;background:white;border-radius:16px}
h1{font-size:26px;margin:0 0 12px}.meta{font-size:14px;color:#526174;margin:12px 0}
select{width:100%;max-width:100%;padding:12px;border:1px solid #cbd5e1;border-radius:8px;font:inherit}
@media(max-width:700px){main{margin:0;padding:12px}}</style></head><body><main>
<h1>Смена аэропорта между перелётами</h1>
<div class="meta">Порог: менее __HOURS__ ч · по датам: 0–__LASTDAY__ календарных дней</div>
<div class="meta">Эпизодов: __EVENTS__ · Пассажиров: __USERS__ · В списке: __SHOWN__ · На проверку: __REVIEW__</div>
<select id="person" aria-label="Пассажир"></select><div id="info" class="meta"></div>
<div class="meta">Зелёный — прилёт · оранжевый — следующий вылет · круг — время записано · ромб — только даты</div>
<div class="meta">Время по записи, без поправки на часовые пояса · Наземная доступность не рассчитана</div>
<div id="graph"></div></main><script>__PLOTLY__</script><script>
const items=__DATA__;const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sel=document.getElementById('person');items.forEach((p,i)=>{const o=document.createElement('option');o.value=i;
o.textContent=`${p.name} (${p.user_id}) · ${p.events.length} эпизодов`;sel.appendChild(o)});
function draw(i){const p=items[i],e=p.events;document.getElementById('info').textContent=`${p.name} · ${p.user_id}`;
let x=[],y=[],ann=[];e.forEach((r,k)=>{x.push(r.x_a,r.x_b,null);y.push(k+.1,k-.1,null);
ann.push({x:new Date((Date.parse(r.x_a)+Date.parse(r.x_b))/2).toISOString(),y:k+.3,text:r.label,showarrow:false,font:{size:12}})});
const traces=[{type:'scatter',mode:'lines',x,y,line:{color:'#b8c4d5',width:2},hoverinfo:'skip',showlegend:false}];
[['a','Прилёт','#168b70',.1],['b','Следующий вылет','#e67e22',-.1]].forEach(([side,label,color,dy])=>{
traces.push({type:'scatter',mode:'markers',x:e.map(r=>r['x_'+side]),y:e.map((r,k)=>k+dy),name:label,
marker:{color,size:12,symbol:e.map(r=>r.precise?'circle':'diamond')},
text:e.map(r=>`${label}: ${esc(r['airport_'+side])} · ${esc(r['city_'+side])}<br>${esc(side==='a'?r.arrival:r.departure)}<br>Рейс: ${esc(r['flight_'+side])}<br>Промежуток: ${r.label}`),hovertemplate:'%{text}<extra></extra>'})});
Plotly.react('graph',traces,{height:Math.max(420,140+e.length*65),margin:{l:155,r:30,t:35,b:70},showlegend:false,
font:{family:'Arial',color:'#172840'},paper_bgcolor:'white',plot_bgcolor:'white',
xaxis:{type:'date',title:'Дата и время по записи',tickformat:'%d.%m.%Y',gridcolor:'#eef2f6'},
yaxis:{tickmode:'array',tickvals:e.map((r,k)=>k),ticktext:e.map((r,k)=>`${k+1}. ${esc(r.airport_a)} → ${esc(r.airport_b)}`),
range:[e.length-.4,-.65],zeroline:false,gridcolor:'#eef2f6'},annotations:ann},
{responsive:true,displaylogo:false,toImageButtonOptions:{filename:'airport_gaps',scale:2}})}
if(items.length){sel.onchange=()=>draw(+sel.value);draw(0)}else{sel.hidden=true;document.getElementById('graph').textContent='Эпизодов с известной датой прилёта и заданными условиями нет.'}
</script></body></html>'''
    replacements={'__HOURS__':str(args.max_days*24),'__LASTDAY__':str(args.max_days-1),
        '__EVENTS__':str(len(data)),'__USERS__':str(total_users),'__SHOWN__':str(len(items)),
        '__REVIEW__':str(stats['review_pairs']),'__PLOTLY__':get_plotlyjs(),'__DATA__':payload}
    for key,value in replacements.items():page=page.replace(key,value)
    target.write_text(page,encoding='utf-8')


def run(args):
    source,out=args.input.resolve(),args.output.resolve()
    if source==out or source in out.parents:
        raise ValueError('Укажите отдельную выходную папку')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Выходная папка не пуста. Укажите новое имя.')
    print('Читаю таблицы и восстанавливаю последовательность перелётов…',flush=True)
    d,excluded,stats=prepare(source)
    accepted,review,all_pairs=classify(d,args.max_days)
    stats.update(qualifying_episodes=len(accepted),qualifying_users=int(accepted.user_id.nunique()),
        review_pairs=len(review),reason_counts=all_pairs.reason.value_counts().to_dict(),
        threshold_days=args.max_days,timezone_correction=False,
        interpretation='Airport changes between adjacent known flights; ground travel feasibility not assessed')
    out.mkdir(parents=True,exist_ok=True)
    save(accepted,out/'airport_gap_candidates.csv')
    save(review,out/'airport_gap_review.csv')
    save(excluded,out/'unknown_departure_dates.csv')
    counts=accepted.groupby(['user_id','name'],sort=False).size().reset_index(name='episodes')
    counts.sort_values('episodes',ascending=False).to_csv(out/'passengers.csv',sep=';',encoding='utf-8-sig',index=False)
    (out/'airport_gap_report.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    create_html(accepted,stats['qualifying_users'],stats,args,out/'06_airport_gaps.html')
    print(f'Готово: {out/"06_airport_gaps.html"}')
    print(f'Эпизодов: {len(accepted):,}; пассажиров: {stats["qualifying_users"]:,}; на проверку: {len(review):,}')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-days',type=int,default=3)
    p.add_argument('--top',type=int,default=100)
    args=p.parse_args()
    if min(args.max_days,args.top)<1:p.error('Пороги должны быть положительными')
    try:run(args)
    except (ValueError,OSError,pd.errors.ParserError) as exc:sys.exit(f'Ошибка: {exc}')


if __name__=='__main__':main()

