"""Поиск профиля и графики. Запуск через run_profile_app.py."""
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from merge_csv import name_token, spelling


DEFAULT_FOLDER = Path(__file__).resolve().parent / "cleaned_v2" / "cleaned_v2"
TABLES = ("users", "flights", "user_flights", "user_loyalty")


def read(folder, name, columns):
    return pd.read_csv(Path(folder) / f"{name}.csv", sep=";", encoding="utf-8-sig",
                       dtype=str, keep_default_na=False, usecols=columns)


# Приводим кириллицу и латиницу к общей форме для поиска.
def normalize_words(value):
    return " ".join(spelling(name_token(t)) for t in re.findall(r"[^\W\d_]+", value, re.UNICODE))


def prepare_users(users):
    if users.user_id.eq("").any() or users.user_id.duplicated().any():
        raise ValueError("В users.csv пустые или повторяющиеся user_id")
    result = users.copy()
    fallback = (result.last_name + " " + result.first_name + " " + result.middle_name).str.replace(r"\s+", " ", regex=True).str.strip()
    result["display_name"] = result.full_name.where(result.full_name.ne(""), fallback)
    result["_search"] = (result.display_name + " " + result.first_name + " " + result.middle_name + " " + result.last_name).map(normalize_words)
    return result


def search_profiles(users, query):
    query = query.strip()
    exact = users.loc[users.user_id.str.upper().eq(query.upper())]
    if not exact.empty:
        return exact.copy()
    tokens = normalize_words(query).split()
    if not tokens or sum(map(len, tokens)) < 2:
        return users.iloc[:0].copy()
    mask = pd.Series(True, index=users.index)
    for token in tokens:
        mask &= users["_search"].str.contains(token, regex=False)
    return users.loc[mask].copy()


def fingerprint(folder):
    return tuple((name, (Path(folder)/f"{name}.csv").stat().st_size,
                  (Path(folder)/f"{name}.csv").stat().st_mtime_ns) for name in TABLES)


@st.cache_data(max_entries=1, show_spinner="Готовлю поиск по именам…")
def load_users(folder, signature):
    return prepare_users(read(folder, "users", ["user_id", "first_name", "middle_name", "last_name",
        "full_name", "birth_date_iso", "sex", "identity_status", "user_needs_review"]))


@st.cache_resource(max_entries=1, show_spinner="Первое чтение готовых таблиц. Следующие профили откроются из памяти…")
def load_relations(folder, signature):
    flights = read(folder, "flights", ["flight_id", "flight_date", "flight_time", "flight_number",
        "departure_airport", "arrival_airport", "departure_city", "arrival_city", "flight_needs_review"])
    if flights.flight_id.eq("").any() or flights.flight_id.duplicated().any():
        raise ValueError("В flights.csv пустые или повторяющиеся flight_id")
    trips = read(folder, "user_flights", ["user_id", "flight_id", "travel_class", "seat", "observation_count", "needs_review"])
    cards = read(folder, "user_loyalty", ["user_id", "program_code", "card_number"])
    if fingerprint(folder) != signature:
        raise ValueError("Файлы изменились во время чтения. Дождись завершения записи и обнови страницу.")
    return flights.set_index("flight_id"), trips.set_index("user_id"), cards.set_index("user_id")


# Достаём перелёты и карты выбранного UID.
def select_profile_data(uid, flights, trips, cards):
    selected = trips.loc[[uid]].copy() if uid in trips.index else trips.iloc[:0].copy()
    selected = selected.reset_index()
    if selected.duplicated(["user_id", "flight_id"]).any():
        raise ValueError("У выбранного профиля повторяется пара user_id + flight_id")
    selected = selected.merge(flights, left_on="flight_id", right_index=True, how="left", validate="many_to_one", indicator=True)
    missing = int(selected["_merge"].ne("both").sum())
    selected = selected.drop(columns="_merge").fillna("")
    selected["date"] = pd.to_datetime(selected.flight_date, format="%Y-%m-%d", errors="coerce")
    selected["route"] = selected.departure_airport.replace("", "не указан") + " → " + selected.arrival_airport.replace("", "не указан")
    selected = selected.sort_values(["date", "flight_time", "flight_id"], na_position="last")
    loyalty = cards.loc[[uid]].copy() if uid in cards.index else cards.iloc[:0].copy()
    loyalty = loyalty.reset_index().drop_duplicates(["user_id", "program_code", "card_number"])
    programs = loyalty.loc[loyalty.program_code.ne("")].groupby("program_code").card_number.nunique().rename_axis("Программа").reset_index(name="Разных карт")
    return selected, programs, missing


def monthly_activity(trips):
    valid = trips.loc[trips.date.notna()]
    if valid.empty:
        return pd.DataFrame(columns=["Месяц", "Перелётов"])
    periods = valid.date.dt.to_period("M")
    counts = periods.value_counts().reindex(pd.period_range(periods.min(), periods.max(), freq="M"), fill_value=0).sort_index()
    return pd.DataFrame({"Месяц": counts.index.astype(str), "Перелётов": counts.values})


def show_profile(user, trips, programs, missing):
    st.subheader(user["display_name"] or "Профиль без имени")
    st.caption(f"user_id: {user['user_id']} · дата рождения: {user['birth_date_iso'] or 'не указана'} · статус: {user['identity_status']}")
    if user["user_needs_review"].lower() == "true":
        st.warning("У профиля есть отметка проверки сопоставления. Его сведения могут быть неполными или противоречивыми.")
    if missing:
        st.warning(f"Для {missing} связей не найден flight_id в flights.csv; сведения о рейсах для них недоступны.")
    a, b, c = st.columns(3)
    a.metric("Перелётов в данных", len(trips))
    known = pd.concat([trips.departure_airport, trips.arrival_airport])
    b.metric("Разных известных аэропортов", int(known.loc[known.ne("")].nunique()))
    c.metric("Программ лояльности", len(programs))
    activity_tab, routes_tab, history_tab, loyalty_tab = st.tabs(["По месяцам", "Маршруты", "История перелётов", "Лояльность"])
    with activity_tab:
        monthly = monthly_activity(trips)
        if monthly.empty:
            st.info("Нет перелётов с распознанной датой.")
        else:
            figure = px.bar(monthly, x="Месяц", y="Перелётов", text="Перелётов", color_discrete_sequence=["#2463a6"])
            figure.update_xaxes(type="category")
            figure.update_yaxes(dtick=1 if monthly["Перелётов"].max() <= 10 else None, rangemode="tozero")
            st.plotly_chart(figure, width="stretch", key="monthly")
        invalid = int(trips.date.isna().sum())
        if invalid:
            st.caption(f"Без распознанной даты: {invalid}. Они не включены в график по месяцам.")
    with routes_tab:
        if trips.empty:
            st.info("У профиля нет связанных перелётов.")
        else:
            routes = trips.groupby("route").size().nlargest(15).sort_values().rename_axis("Маршрут").reset_index(name="Перелётов")
            figure = px.bar(routes, x="Перелётов", y="Маршрут", orientation="h", text="Перелётов", color_discrete_sequence=["#398579"])
            figure.update_xaxes(dtick=1 if routes["Перелётов"].max() <= 10 else None, rangemode="tozero")
            st.plotly_chart(figure, width="stretch", key="routes")
    with history_tab:
        dated = trips.loc[trips.date.notna()].copy()
        if not dated.empty:
            dated["Класс"] = dated.travel_class.replace("", "Не указан")
            st.plotly_chart(px.scatter(dated, x="date", y="route", color="Класс", hover_data=["flight_number", "flight_time"],
                                      labels={"date": "Дата вылета", "route": "Маршрут", "flight_number": "Рейс", "flight_time": "Время из источника"}), width="stretch", key="timeline")
        st.dataframe(trips[["flight_date", "flight_time", "flight_number", "departure_airport", "arrival_airport", "travel_class", "seat", "observation_count", "needs_review"]], hide_index=True, width="stretch")
    with loyalty_tab:
        if programs.empty:
            st.info("В этой выгрузке программы лояльности не найдены.")
        else:
            figure = px.bar(programs, x="Программа", y="Разных карт", text="Разных карт", color_discrete_sequence=["#9673a6"])
            figure.update_yaxes(dtick=1 if programs["Разных карт"].max() <= 10 else None, rangemode="tozero")
            st.plotly_chart(figure, width="stretch", key="programs")
            st.dataframe(programs, hide_index=True, width="stretch")
    st.caption("Показаны наблюдаемые записи одного выбранного user_id. Совпадающие имена не объединяются. observation_count — число исходных наблюдений, а не число перелётов. Нулевое число означает отсутствие записей в этой выгрузке.")


def main():
    st.set_page_config(page_title="Перелёты выбранного профиля", page_icon="✈", layout="wide")
    st.set_option("client.toolbarMode", "minimal")
    st.title("Перелёты выбранного профиля")
    st.write("Найди профиль по имени или user_id, затем открой его графики.")
    folder_text = st.sidebar.text_input("Папка четырёх CSV", value=str(DEFAULT_FOLDER))
    folder = str(Path(folder_text).expanduser().resolve())
    if st.sidebar.button("Перечитать файлы"):
        load_users.clear()
        load_relations.clear()
    try:
        signature = fingerprint(folder)
        users = load_users(folder, signature)
    except (OSError, ValueError) as exc:
        st.error(f"Не удалось прочитать данные: {exc}")
        st.stop()
    with st.form("search_form"):
        query = st.text_input("Имя, фамилия или user_id", placeholder="Например: Иван Иванов или U000045027")
        submitted = st.form_submit_button("Найти", type="primary")
    if submitted:
        st.session_state["search_query"] = query.strip()
    query = st.session_state.get("search_query", "")
    if not query:
        st.info("Можно вводить кириллицей или латиницей. Поиск не меняет решения мерджа.")
        st.stop()
    matches = search_profiles(users, query)
    if matches.empty:
        st.info("Совпадений нет. Попробуй часть фамилии или точный user_id.")
        st.stop()
    if len(matches) > 100:
        st.warning(f"Найдено профилей: {len(matches)}. Добавь имя или введи точный user_id.")
        st.stop()
    indexed = matches.set_index("user_id")
    def label(uid):
        u = indexed.loc[uid]
        return f"{u['display_name'] or 'Без имени'} — {uid} — ДР: {u['birth_date_iso'] or 'нет данных'}"
    st.caption(f"Найдено профилей: {len(matches)}. Одинаковое имя не означает одного человека.")
    uid = st.selectbox("Выбери профиль", options=matches.user_id.tolist(), format_func=label)
    try:
        relations = load_relations(folder, signature)
        trips, programs, missing = select_profile_data(uid, *relations)
    except (OSError, ValueError) as exc:
        st.error(f"Не удалось построить карточку: {exc}")
        st.stop()
    show_profile({**indexed.loc[uid].to_dict(), "user_id": uid}, trips, programs, missing)


if __name__ == "__main__":
    main()
