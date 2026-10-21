#!/usr/bin/env python3
import argparse
import csv
import html
import itertools
import json
import re
import sqlite3
import sys
import unicodedata
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


def clean_name(value):
    value = unicodedata.normalize('NFKD', str(value)).casefold()
    return ''.join(c for c in value if c.isalpha() and not unicodedata.combining(c)).replace('y', 'i')


def close_name(a, b):
    if not a or not b:
        return False
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = next((i for i in range(len(a)) if a[i] != b[i]), len(a))
    return a[i:] == b[i + 1:]


def identity_reason(a, b):
    invalid = {'', 'none', 'nan', 'null', 'unknown', 'n/a', '-'}
    pa, pb = a.get('passport_key', ''), b.get('passport_key', '')
    if pa.casefold() not in invalid and pa == pb:
        return 'same_passport'
    if pa.casefold() not in invalid and pb.casefold() not in invalid and pa != pb:
        return ''
    da, db = a.get('birth_date_iso', ''), b.get('birth_date_iso', '')
    if da and db and da != db:
        return ''
    fa, fb = clean_name(a.get('first_name', '')), clean_name(b.get('first_name', ''))
    la, lb = clean_name(a.get('last_name', '')), clean_name(b.get('last_name', ''))
    if fa and fb and la and lb:
        if close_name(fa, fb) and close_name(la, lb):
            return 'similar_name_compatible_identity'
    elif close_name(clean_name(a['name']), clean_name(b['name'])):
        return 'similar_full_name_compatible_identity'
    return ''


def load_data(folder):
    users = read_table(folder / 'users.csv', ['user_id'],
        ['first_name', 'middle_name', 'last_name', 'full_name', 'passport_key', 'birth_date_iso'])
    flights = read_table(folder / 'flights.csv',
        ['flight_id', 'flight_date', 'departure_airport', 'arrival_airport'],
        ['flight_number', 'departure_city', 'arrival_city'])
    links = read_table(folder / 'user_flights.csv', ['user_id', 'flight_id'])
    for frame, key in [(users, 'user_id'), (flights, 'flight_id')]:
        if frame[key].eq('').any() or frame[key].duplicated().any():
            raise ValueError(f'{key} должен быть непустым и уникальным')
    if not links.user_id.isin(users.user_id).all() or not links.flight_id.isin(flights.flight_id).all():
        raise ValueError('user_flights содержит ссылки на отсутствующих пассажиров или рейсы')
    users = users.sort_values('user_id').reset_index(drop=True)
    users['u'] = np.arange(len(users))
    users['name'] = ''
    if 'full_name' in users:
        users['name'] = users.full_name
    split = pd.Series('', index=users.index, dtype='string')
    for col in ['first_name', 'middle_name', 'last_name']:
        if col in users:
            split = split.str.cat(users[col], sep=' ').str.strip()
    users['name'] = users.name.mask(users.name.eq(''), split)
    users['name'] = users.name.mask(users.name.eq(''), users.user_id)
    stats = {'input_links': len(links), 'duplicate_links_removed': int(links.duplicated().sum())}
    events = links.drop_duplicates().merge(flights, on='flight_id', validate='many_to_one')
    dates = pd.to_datetime(events.flight_date, format='%Y-%m-%d', errors='coerce')
    dep = events.departure_airport.str.upper()
    arr = events.arrival_airport.str.upper()
    valid_route = dep.str.fullmatch('[A-Z]{3}').fillna(False) & arr.str.fullmatch('[A-Z]{3}').fillna(False) & dep.ne(arr)
    stats['excluded_invalid_date'] = int(dates.isna().sum())
    stats['excluded_invalid_or_same_airport_route'] = int((~valid_route).sum())
    mask = dates.notna() & valid_route
    events = events.loc[mask].copy()
    events['day'] = dates.loc[mask].to_numpy(dtype='datetime64[D]').astype(np.int64)
    events['route'] = dep.loc[mask] + ' → ' + arr.loc[mask]
    events['u'] = events.user_id.map(users.set_index('user_id').u).astype(int)
    route_codes, route_labels = pd.factorize(events.route, sort=True)
    events['r'] = route_codes
    number = events.get('flight_number', pd.Series('', index=events.index)).str.upper()
    flight_key = events.flight_id.astype(str)
    has_number = number.ne('')
    flight_key = flight_key.mask(has_number,
        events.flight_date + '|' + events.route + '|' + number)
    events['f'] = pd.factorize(flight_key, sort=True)[0]
    stats['valid_event_records'] = len(events)
    return users, events, list(route_labels), stats


def write_matches(con, left, right, same, chunk_size):
    inserted = 0
    keys = ['f'] if same else ['r', 'join_day']
    sql = 'INSERT INTO matches VALUES (?,?,?,?,?,?) ON CONFLICT(a,b,da,r,db) DO UPDATE SET same=max(same,excluded.same)'
    bucket_key = 'f' if same else 'r'
    right_buckets = {int(k): g for k, g in right.groupby(right[bucket_key] % 128, sort=False)}
    processed = 0
    for bucket, source in left.groupby(left[bucket_key] % 128, sort=False):
        destination = right_buckets.get(int(bucket))
        if destination is None:
            continue
        for start in range(0, len(source), chunk_size):
            joined = source.iloc[start:start + chunk_size].merge(destination, on=keys, suffixes=('_a', '_b'))
            joined = joined.loc[joined.u_a.ne(joined.u_b)]
            if len(joined):
                if same:
                    values = joined[['u_a', 'u_b', 'day_a', 'day_b', 'r_a']].drop_duplicates()
                else:
                    values = joined[['u_a', 'u_b', 'day_a', 'day_b', 'r']].drop_duplicates()
                con.executemany(sql, ((*map(int, row), int(same)) for row in values.itertuples(index=False, name=None)))
                inserted += len(values)
        previous = processed
        processed += len(source)
        if processed // 100000 != previous // 100000:
            con.commit()
            print(f'  Проверено записей: {processed:,}/{len(left):,}', flush=True)
    con.commit()
    return inserted


def collect_matches(con, events, args):
    con.execute('CREATE TABLE matches(a INTEGER,b INTEGER,da INTEGER,db INTEGER,r INTEGER,same INTEGER, PRIMARY KEY(a,b,da,r,db)) WITHOUT ROWID')
    same = events[['u', 'day', 'r', 'f']].drop_duplicates()
    daily = events[['u', 'day', 'r']].drop_duplicates()
    print('Ищу совместные рейсы…', flush=True)
    write_matches(con, same, same, True, args.chunk_size)
    right = daily.assign(join_day=daily.day)
    for lag in range(1, args.max_lag + 1):
        print(f'Ищу повторение маршрута через {lag} дн.…', flush=True)
        left = daily.assign(join_day=daily.day + lag)
        write_matches(con, left, right, False, args.chunk_size)


def one_to_one(rows):
    used_a, used_b, selected = set(), set(), []
    for da, db, route, same in sorted(rows, key=lambda x: (x[0], x[1], x[2], -x[3])):
        ka, kb = (da, route), (db, route)
        if ka not in used_a and kb not in used_b:
            used_a.add(ka)
            used_b.add(kb)
            selected.append((da, db, route, same))
    return selected


PAIR_COLS = ['person_a_id', 'person_a_name', 'person_b_id', 'person_b_name',
    'matched_routes_dates', 'distinct_dates_a', 'distinct_dates_b', 'distinct_routes',
    'same_flight_matches', 'delayed_matches', 'median_lag_days',
    'coverage_a_pct', 'coverage_b_pct', 'total_routes_dates_a', 'total_routes_dates_b',
    'identity_review', 'direction']
MATCH_COLS = ['person_a_id', 'person_b_id', 'date_a', 'date_b', 'route', 'lag_days', 'same_flight', 'identity_review']


def day_string(day):
    return str(np.datetime64(int(day), 'D'))


def summarize(con, users, events, routes, args, out):
    con.execute('CREATE TEMP TABLE eligible AS SELECT a,b FROM matches GROUP BY a,b HAVING count(DISTINCT da)>=? AND count(DISTINCT db)>=?', (args.min_matches, args.min_matches))
    con.execute('CREATE UNIQUE INDEX eligible_ab ON eligible(a,b)')
    total = events[['u', 'day', 'r']].drop_duplicates().groupby('u').size().to_dict()
    profiles = users.to_dict('records')
    pairs, review, graph_candidates = [], [], []
    stream = con.execute('SELECT m.a,m.b,m.da,m.db,m.r,m.same FROM eligible e JOIN matches m ON m.a=e.a AND m.b=e.b ORDER BY m.a,m.b,m.da,m.db,m.r')
    with (out / 'matching_flights.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(MATCH_COLS)
        for (a, b), records in itertools.groupby(stream, key=lambda r: (r[0], r[1])):
            matches = one_to_one([r[2:] for r in records])
            n = len(matches)
            na, nb = len({m[0] for m in matches}), len({m[1] for m in matches})
            nr = len({m[2] for m in matches})
            if min(na, nb) < args.min_matches or nr < args.min_routes:
                continue
            shared = sum(m[3] for m in matches)
            if shared == n and a > b:
                continue
            pa, pb = profiles[a], profiles[b]
            reason = identity_reason(pa, pb)
            lags = [m[1] - m[0] for m in matches]
            record = dict(zip(PAIR_COLS, [pa['user_id'], pa['name'], pb['user_id'], pb['name'],
                n, na, nb, nr, shared, n-shared, float(np.median(lags)),
                round(100*n/total[a], 2), round(100*n/total[b], 2), total[a], total[b],
                reason, 'same_flights_only' if shared == n else 'A_then_B']))
            (review if reason else pairs).append(record)
            details = []
            for da, db, r, is_same in matches:
                writer.writerow([pa['user_id'], pb['user_id'], day_string(da), day_string(db),
                                 routes[r], db-da, bool(is_same), reason])
                details.append({'date_a': day_string(da), 'date_b': day_string(db),
                                'route': routes[r], 'lag': db-da, 'same': bool(is_same)})
            if not reason:
                graph_candidates.append({'pair': record, 'matches': details})
                graph_candidates.sort(key=rank, reverse=True)
                del graph_candidates[args.top:]
    frame = pd.DataFrame(pairs, columns=PAIR_COLS)
    frame = frame.sort_values(['coverage_a_pct','matched_routes_dates','distinct_routes'], ascending=False)
    frame.to_csv(out/'pairs.csv', sep=';', encoding='utf-8-sig', index=False)
    pd.DataFrame(review, columns=PAIR_COLS).to_csv(out/'identity_review_pairs.csv', sep=';', encoding='utf-8-sig', index=False)
    return frame, review, graph_candidates


def rank(item):
    p = item['pair']
    return p['coverage_a_pct'], p['matched_routes_dates'], p['distinct_routes']


def build_html(items, count, review_count, args, target):
    payload = json.dumps(items, ensure_ascii=False).replace('<', '\\u003c')
    subtitle = (f'≥{args.min_matches} совпадения в разные даты · тот же рейс или через 1–{args.max_lag} дн. · '
                f'≥{args.min_routes} направление') if args.max_lag else f'≥{args.min_matches} совпадения в разные даты · только тот же рейс'
    page = '''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Повторение маршрутов</title>
<style>body{margin:0;background:#f3f5f8;font:15px Arial,sans-serif;color:#172840}
main{max-width:1400px;margin:24px auto;padding:24px;background:white;border-radius:16px}
h1{font-size:26px;margin:0 0 12px}.meta{color:#526174;font-size:14px;margin:10px 0}
select{max-width:100%;width:100%;font:inherit;padding:12px;border:1px solid #cbd5e1;border-radius:8px}
.metrics{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}.metric{background:#f1f5f9;padding:12px;border-radius:8px}
.metric b{font-size:22px;display:block;margin-bottom:5px}#graph{min-height:400px}
@media(max-width:700px){main{margin:0;padding:12px}} </style></head><body><main>
<h1>Повторение маршрутов</h1><div class="meta">__SUBTITLE__</div>
<div class="meta">Пар: __COUNT__ · На графике: __SHOWN__ · На проверку личности: __REVIEW__</div>
<select id="pair" aria-label="Выбрать пару"></select><div class="metrics" id="metrics"></div>
<div class="meta" id="names"></div>
<div class="meta">Оранжевый — А · синий — Б · линия — совпадение маршрута · подпись — задержка Б</div>
<div id="graph"></div></main><script>__PLOTLY__</script><script>
const items=__DATA__;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const select=document.getElementById('pair');
items.forEach((item,i)=>{const p=item.pair,o=document.createElement('option');o.value=i;
o.textContent=`${i+1}. ${p.person_a_name} ${p.direction==='same_flights_only'?'↔':'→'} ${p.person_b_name} · ${p.matched_routes_dates} совп. · охват А ${p.coverage_a_pct}%`;select.appendChild(o)});
function draw(index){
const item=items[index],p=item.pair,m=item.matches;
document.getElementById('names').textContent=`А: ${p.person_a_name} (${p.person_a_id}) · Б: ${p.person_b_name} (${p.person_b_id})`;
const metrics=[[p.matched_routes_dates,'совпадений'],[p.distinct_routes,'направлений'],
[p.coverage_a_pct+'%','охват поездок А'],[p.same_flight_matches,'на одном рейсе'],[p.delayed_matches,'с задержкой']];
document.getElementById('metrics').innerHTML=metrics.map(([v,t])=>`<div class="metric"><b>${esc(v)}</b>${esc(t)}</div>`).join('');
let linesX=[],linesY=[],annotations=[];
m.forEach((r,i)=>{linesX.push(r.date_a,r.date_b,null);linesY.push(i+.12,i-.12,null);
annotations.push({x:new Date((Date.parse(r.date_a)+Date.parse(r.date_b))/2).toISOString(),y:i+.28,
text:r.same?'тот же рейс':`+${r.lag} дн.`,showarrow:false,font:{size:11,color:'#526174'}})});
const traces=[{type:'scatter',mode:'lines',x:linesX,y:linesY,line:{color:'#b8c4d5',width:2},hoverinfo:'skip',showlegend:false}];
[['a','А','#e67e22',.12],['b','Б','#2875c7',-.12]].forEach(([key,label,color,offset])=>{
traces.push({type:'scatter',mode:'markers',name:label,x:m.map(r=>r['date_'+key]),y:m.map((r,i)=>i+offset),
marker:{color,size:11},text:m.map(r=>`${label}: ${esc(key==='a'?p.person_a_name:p.person_b_name)}<br>${esc(r.route)}<br>${r['date_'+key]}<br>${r.same?'Тот же рейс':`Б через ${r.lag} дн.`}`),hovertemplate:'%{text}<extra></extra>'})});
Plotly.react('graph',traces,{template:'plotly_white',height:Math.max(420,130+m.length*55),
margin:{l:155,r:35,t:30,b:70},font:{family:'Arial',color:'#172840'},showlegend:false,
xaxis:{type:'date',title:'Дата вылета',tickformat:'%d.%m.%Y',showgrid:true},
yaxis:{tickmode:'array',tickvals:m.map((r,i)=>i),ticktext:m.map((r,i)=>`${i+1}. ${esc(r.route)}`),
range:[m.length-.4,-.65],zeroline:false},annotations},
{responsive:true,displaylogo:false,toImageButtonOptions:{filename:'route_matches',scale:2}});
}
if(items.length){select.onchange=()=>draw(Number(select.value));draw(0)}else{
select.hidden=true;document.getElementById('graph').textContent='Пар с заданными условиями не найдено.';}
</script></body></html>'''
    for key, value in [('__SUBTITLE__', subtitle), ('__COUNT__', str(count)), ('__SHOWN__', str(len(items))),
                       ('__REVIEW__', str(review_count)), ('__PLOTLY__', get_plotlyjs()), ('__DATA__', payload)]:
        page = page.replace(key, value)
    target.write_text(page, encoding='utf-8')


def run(args):
    source, out = args.input.resolve(), args.output.resolve()
    if source == out or source in out.parents:
        raise ValueError('Выходная папка должна находиться отдельно от входной')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Выходная папка не пуста. Укажите новое имя папки.')
    print('Читаю таблицы…', flush=True)
    users, events, routes, stats = load_data(source)
    out.mkdir(parents=True, exist_ok=True)
    database = out / 'working_matches.sqlite'
    con = sqlite3.connect(database)
    try:
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('PRAGMA cache_size=-64000')
        collect_matches(con, events, args)
        print('Проверяю повторения, охват и возможные дубли людей…', flush=True)
        pairs, review, items = summarize(con, users, events, routes, args, out)
    finally:
        con.close()
    stats.update(pairs=len(pairs), identity_review_pairs=len(review), html_pairs=len(items),
        parameters={k:v for k,v in vars(args).items() if k not in ['input','output']},
        route_definition='Directed airport pair; missing, invalid or equal airports excluded',
        lag_definition='Calendar departure dates; different flights on same date excluded',
        coverage_definition='Matched unique route/dates divided by all valid route/dates of A',
        matching='Chronological one-to-one matching per route/date within each directed pair',
        ranking='Coverage A descending, then matching route/dates, then distinct routes',
        identity='Same document or similar names with no conflicting document/DOB sent to review; no merging',
        interpretation='Repeated route coincidences; no inference of intent')
    (out/'followers_report.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    build_html(items, len(pairs), len(review), args, out/'05_followers.html')
    if not args.keep_database:
        database.unlink(missing_ok=True)
    print(f'Готово: {out / "05_followers.html"}')
    print(f'Пар: {len(pairs):,}; на проверку личности: {len(review):,}; показано: {len(items):,}')


def main():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-lag', type=int, default=3)
    parser.add_argument('--min-matches', type=int, default=2)
    parser.add_argument('--min-routes', type=int, default=1)
    parser.add_argument('--top', type=int, default=50)
    parser.add_argument('--chunk-size', type=int, default=2000)
    parser.add_argument('--keep-database', action='store_true')
    args=parser.parse_args()
    if args.max_lag < 0 or args.min_matches < 2 or min(args.min_routes,args.top,args.chunk_size) < 1:
        parser.error('Проверьте пороги: минимум два совпадения, задержка >=0, остальные значения >=1')
    try:
        run(args)
    except (ValueError,OSError,pd.errors.ParserError,sqlite3.Error) as exc:
        sys.exit(f'Ошибка: {exc}')


if __name__=='__main__':
    main()

