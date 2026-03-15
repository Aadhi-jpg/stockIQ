# -*- coding: utf-8 -*-
# ============================================================
#   StockIQ -- Stock Analysis Dashboard
#   Author  : Aadhithya Rajesh
#   Run with: python dashboard.py
#   Then open: http://127.0.0.1:8050
# ============================================================

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Aadhi's StockIQ - Stock Analysis Dashboard"

DARK = {
    "bg":       "#0f1117",
    "sidebar":  "#1a1d27",
    "card":     "#1e2130",
    "border":   "#2e3250",
    "text":     "#e8eaf6",
    "subtext":  "#8b90a0",
    "accent":   "#7c6af7",
    "accent2":  "#5dcaa5",
    "positive": "#5dcaa5",
    "negative": "#f07b5d",
    "plot_bg":  "#1e2130",
    "grid":     "#2e3250",
}
LIGHT = {
    "bg":       "#f5f6fa",
    "sidebar":  "#ffffff",
    "card":     "#ffffff",
    "border":   "#e0e3ef",
    "text":     "#1a1d27",
    "subtext":  "#6b7080",
    "accent":   "#534AB7",
    "accent2":  "#0F6E56",
    "positive": "#0F6E56",
    "negative": "#993C1D",
    "plot_bg":  "#ffffff",
    "grid":     "#e8eaf6",
}

POPULAR_STOCKS = [
    {"label": "TCS (India)",          "value": "TCS.NS"},
    {"label": "Infosys (India)",      "value": "INFY.NS"},
    {"label": "Reliance (India)",     "value": "RELIANCE.NS"},
    {"label": "HDFC Bank (India)",    "value": "HDFCBANK.NS"},
    {"label": "ITC (India)",          "value": "ITC.NS"},
    {"label": "Wipro (India)",        "value": "WIPRO.NS"},
    {"label": "SBI (India)",          "value": "SBIN.NS"},
    {"label": "Tata Motors (India)",  "value": "TATAMOTORS.NS"},
    {"label": "Sun Pharma (India)",   "value": "SUNPHARMA.NS"},
    {"label": "Apple (US)",           "value": "AAPL"},
    {"label": "Tesla (US)",           "value": "TSLA"},
    {"label": "Google (US)",          "value": "GOOGL"},
    {"label": "Microsoft (US)",       "value": "MSFT"},
    {"label": "Amazon (US)",          "value": "AMZN"},
    {"label": "Nvidia (US)",          "value": "NVDA"},
    {"label": "Meta (US)",            "value": "META"},
    {"label": "Bitcoin",              "value": "BTC-USD"},
    {"label": "Ethereum",             "value": "ETH-USD"},
    {"label": "Nifty 50 (Index)",     "value": "^NSEI"},
    {"label": "S&P 500 (Index)",      "value": "^GSPC"},
]


def download(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def safe_float(val):
    if hasattr(val, 'iloc'):
        return float(val.iloc[0])
    return float(val)


def label_style():
    return {"fontSize": "11px", "fontWeight": "600",
            "textTransform": "uppercase", "letterSpacing": "0.08em",
            "marginBottom": "6px", "display": "block"}


app.layout = html.Div(id="root", children=[
    dcc.Store(id="theme-store", data="dark"),
    html.Script(src="https://www.googletagmanager.com/gtag/js?id=G-2P6KEP6G1M", async_=True),
    html.Script(children="""
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-2P6KEP6G1M');
    """),

    # SIDEBAR
    html.Div(id="sidebar", children=[
        html.Div("Aadhi's StockIQ", style={"fontSize": "20px", "fontWeight": "700", "marginBottom": "4px"}),
        html.Div("Stock Analysis Dashboard", style={"fontSize": "11px", "marginBottom": "24px"}),
        html.Button("Toggle Dark / Light", id="theme-btn", n_clicks=0),
        html.Hr(id="hr1"),

        html.Label("Stock Symbol", style=label_style()),
        dcc.Input(id="stock-input", type="text", value="TCS.NS",
                  placeholder="e.g. TCS.NS or AAPL", debounce=False),
        html.Br(),

        html.Label("Quick Select", style={**label_style(), "marginTop": "14px"}),
        dcc.Dropdown(id="quick-select", options=POPULAR_STOCKS,
                     placeholder="Pick a stock...", clearable=True,
                     style={"fontSize": "13px", "marginTop": "4px", "marginBottom": "14px"}),

        html.Label("Start Date", style=label_style()),
        dcc.Input(id="start-date", type="text", value="2022-01-01",
                  placeholder="YYYY-MM-DD", debounce=False),
        html.Br(),

        html.Label("End Date", style={**label_style(), "marginTop": "14px"}),
        dcc.Input(id="end-date", type="text", value="2026-03-15",
                  placeholder="YYYY-MM-DD", debounce=False),

        html.Hr(id="hr2"),

        html.Label("Predict Ahead (years)", style=label_style()),
        dcc.Slider(id="pred-slider", min=1, max=10, step=1, value=6,
                   marks={i: str(i) for i in range(1, 11)},
                   tooltip={"placement": "bottom"}),
        html.Div(style={"marginBottom": "20px"}),

        html.Label("Moving Avg Window (days)", style=label_style()),
        dcc.Slider(id="ma-slider", min=10, max=200, step=10, value=50,
                   marks={10: "10", 50: "50", 100: "100", 200: "200"},
                   tooltip={"placement": "bottom"}),
        html.Div(style={"marginBottom": "24px"}),

        html.Button("Run Analysis", id="run-btn", n_clicks=0),
        html.Hr(id="hr3"),

        html.Label("Compare With (optional)", style=label_style()),
        dcc.Input(id="compare-input", type="text", value="",
                  placeholder="e.g. INFY.NS", debounce=False),
    ]),

    # MAIN
    html.Div(id="main-content", children=[
        html.Div([
            html.Div(id="page-title",   style={"fontSize": "20px", "fontWeight": "600"}),
            html.Div(id="status-badge"),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "20px"}),

        html.Div(id="metric-cards", style={
            "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "12px", "marginBottom": "20px"
        }),

        dcc.Tabs(id="tabs", value="charts", children=[
            dcc.Tab(label="Charts",     value="charts"),
            dcc.Tab(label="Comparison", value="compare"),
            dcc.Tab(label="News",       value="news"),
        ], style={"marginBottom": "16px"}),

        html.Div(id="tab-content"),
    ]),
])


@app.callback(
    Output("theme-store", "data"),
    Input("theme-btn", "n_clicks"),
    State("theme-store", "data"),
)
def toggle_theme(n, current):
    if n and n > 0:
        return "light" if current == "dark" else "dark"
    return current


@app.callback(
    Output("root",          "style"),
    Output("sidebar",       "style"),
    Output("main-content",  "style"),
    Output("theme-btn",     "style"),
    Output("stock-input",   "style"),
    Output("start-date",    "style"),
    Output("end-date",      "style"),
    Output("compare-input", "style"),
    Output("run-btn",       "style"),
    Output("hr1", "style"),
    Output("hr2", "style"),
    Output("hr3", "style"),
    Input("theme-store", "data"),
)
def apply_theme(theme):
    C = DARK if theme == "dark" else LIGHT
    root    = {"display": "flex", "minHeight": "100vh",
               "fontFamily": "Inter, Segoe UI, sans-serif",
               "backgroundColor": C["bg"], "color": C["text"]}
    sidebar = {"width": "240px", "minWidth": "240px",
               "backgroundColor": C["sidebar"],
               "borderRight": f"1px solid {C['border']}",
               "padding": "24px 18px", "display": "flex",
               "flexDirection": "column", "minHeight": "100vh", "overflowY": "auto"}
    main    = {"flex": "1", "padding": "28px 32px",
               "backgroundColor": C["bg"], "overflowY": "auto"}
    btn     = {"width": "100%", "padding": "8px", "borderRadius": "8px",
               "border": f"1px solid {C['border']}", "cursor": "pointer",
               "fontSize": "12px", "fontWeight": "500", "marginBottom": "20px",
               "backgroundColor": C["card"], "color": C["text"]}
    inp     = {"width": "100%", "padding": "9px 12px", "borderRadius": "8px",
               "border": f"1px solid {C['border']}", "fontSize": "13px",
               "marginTop": "4px", "marginBottom": "4px", "outline": "none",
               "backgroundColor": C["bg"], "color": C["text"],
               "boxSizing": "border-box"}
    run     = {"width": "100%", "padding": "12px", "borderRadius": "10px",
               "border": "none", "fontSize": "14px", "fontWeight": "700",
               "cursor": "pointer", "backgroundColor": C["accent"], "color": "#ffffff"}
    hr      = {"borderColor": C["border"], "margin": "16px 0"}
    return root, sidebar, main, btn, inp, inp, inp, inp, run, hr, hr, hr


@app.callback(
    Output("stock-input", "value"),
    Input("quick-select", "value"),
    prevent_initial_call=True,
)
def fill_from_quick(val):
    return val if val else dash.no_update


@app.callback(
    Output("page-title",   "children"),
    Output("status-badge", "children"),
    Output("metric-cards", "children"),
    Output("tab-content",  "children"),
    Input("run-btn",       "n_clicks"),
    Input("tabs",          "value"),
    State("stock-input",   "value"),
    State("start-date",    "value"),
    State("end-date",      "value"),
    State("pred-slider",   "value"),
    State("ma-slider",     "value"),
    State("compare-input", "value"),
    State("theme-store",   "data"),
)
def run_analysis(n_clicks, tab, symbol, start, end,
                 pred_years, ma_win, compare_sym, theme):
    C = DARK if theme == "dark" else LIGHT

    if not symbol:
        return ("StockIQ", "", [],
                html.Div("Enter a stock symbol and click Run Analysis",
                         style={"color": C["subtext"], "marginTop": "60px",
                                "textAlign": "center", "fontSize": "15px"}))

    df = download(symbol, start, end)
    if df.empty:
        return (symbol, "", [],
                html.Div("No data found. Check the symbol.",
                         style={"color": C["negative"]}))

    days       = len(df)
    highest    = safe_float(df['High'].max())
    lowest     = safe_float(df['Low'].min())
    avg        = safe_float(df['Close'].mean())
    close_vals = df['Close'].values.flatten()
    last_price = float(close_vals[-1])
    prev_price = float(close_vals[-2])
    change_pct = ((last_price - prev_price) / prev_price) * 100
    chg_color  = C["positive"] if change_pct >= 0 else C["negative"]
    chg_sign   = "+" if change_pct >= 0 else ""

    def card(label, value, sub, sub_color=None):
        return html.Div([
            html.Div(label, style={"fontSize": "10px", "textTransform": "uppercase",
                                   "letterSpacing": "0.07em", "color": C["subtext"]}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "600",
                                   "color": C["text"], "margin": "4px 0"}),
            html.Div(sub,   style={"fontSize": "11px",
                                   "color": sub_color or C["subtext"]}),
        ], style={"backgroundColor": C["card"], "borderRadius": "10px",
                  "padding": "14px 16px", "border": f"1px solid {C['border']}"})

    cards = [
        card("Last Price", f"{last_price:,.2f}", f"{chg_sign}{change_pct:.2f}% today", chg_color),
        card("Highest",    f"{highest:,.2f}",    f"Over {days} days"),
        card("Lowest",     f"{lowest:,.2f}",     f"Over {days} days"),
        card("Average",    f"{avg:,.2f}",        "Mean close price"),
    ]

    status = html.Span(f"{days} trading days loaded -- {symbol}", style={
        "backgroundColor": C["card"], "color": C["accent2"],
        "padding": "5px 14px", "borderRadius": "20px",
        "fontSize": "12px", "border": f"1px solid {C['border']}"
    })

    def fig_base(fig, title_text):
        fig.update_layout(
            title=title_text,
            paper_bgcolor=C["card"], plot_bgcolor=C["plot_bg"],
            font_color=C["text"], font_family="Inter, Segoe UI, sans-serif",
            title_font_size=14, margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=C["grid"], linecolor=C["border"]),
            yaxis=dict(gridcolor=C["grid"], linecolor=C["border"]),
        )
        return fig

    def wrap(graph, note=None):
        kids = [dcc.Graph(figure=graph, config={"displayModeBar": False})]
        if note:
            kids.append(html.Div(note, style={"fontSize": "11px",
                                              "color": C["subtext"],
                                              "padding": "0 12px 10px"}))
        return html.Div(kids, style={
            "backgroundColor": C["card"], "borderRadius": "12px",
            "border": f"1px solid {C['border']}", "padding": "8px", "flex": "1"
        })

    # ── CHARTS TAB ────────────────────────────────────────────
    if tab == "charts":
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df.index, y=close_vals, mode='lines', name='Close Price',
            line=dict(color=C["accent"], width=1.8),
            fill='tozeroy', fillcolor="rgba(124,106,247,0.08)"
        ))
        fig1 = fig_base(fig1, f"{symbol} -- Closing Price")

        df['MA']    = df['Close'].rolling(window=ma_win).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=close_vals, mode='lines',
                                  name='Daily Price',
                                  line=dict(color=C["accent"], width=1), opacity=0.5))
        fig2.add_trace(go.Scatter(x=df.index, y=df['MA'].values.flatten(),
                                  mode='lines', name=f'{ma_win}-Day MA',
                                  line=dict(color="#EF9F27", width=2.5)))
        if days > 200:
            fig2.add_trace(go.Scatter(x=df.index, y=df['MA200'].values.flatten(),
                                      mode='lines', name='200-Day MA',
                                      line=dict(color=C["negative"], width=2, dash='dash')))
        fig2 = fig_base(fig2, f"{symbol} -- Moving Averages")

        df['Month']  = df.index.month
        df['Return'] = df['Close'].pct_change() * 100
        monthly      = df.groupby('Month')['Return'].mean()
        month_names  = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']
        avail        = [month_names[i-1] for i in monthly.index]
        bar_colors   = [C["positive"] if x >= 0 else C["negative"]
                        for x in monthly.values]
        best_month   = month_names[int(monthly.idxmax()) - 1]
        worst_month  = month_names[int(monthly.idxmin()) - 1]
        fig3 = go.Figure(go.Bar(x=avail, y=monthly.values.flatten(),
                                marker_color=bar_colors))
        fig3 = fig_base(fig3, f"{symbol} -- Average Monthly Return (%)")

        df_close  = df['Close'].dropna()
        carr      = df_close.values.flatten()
        x         = np.array(range(len(carr))).reshape(-1, 1)
        model     = LinearRegression()
        model.fit(x, carr)
        fut_days  = pred_years * 252
        fut_x     = np.array(range(len(carr) + fut_days)).reshape(-1, 1)
        fut_pred  = model.predict(fut_x)
        last_date = df_close.index[-1]
        fut_dates = pd.date_range(last_date, periods=fut_days + 1, freq='B')[1:]
        all_dates = list(df_close.index) + list(fut_dates)
        pred_val  = round(float(fut_pred[-1]), 2)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df_close.index, y=carr, mode='lines', name='Actual',
            line=dict(color=C["accent"], width=1.5)
        ))
        fig4.add_trace(go.Scatter(
            x=all_dates, y=fut_pred, mode='lines', name='Prediction',
            line=dict(color=C["negative"], width=2.5, dash='dash')
        ))
        # Vertical line marking prediction start
        fig4.add_trace(go.Scatter(
            x=[last_date, last_date],
            y=[float(min(carr)), float(max(carr))],
            mode='lines', name='Prediction start',
            line=dict(color=C["subtext"], width=1, dash='dot'),
            showlegend=False
        ))
        fig4 = fig_base(fig4, f"{symbol} -- Prediction ({pred_years} yrs ahead)")

        content = html.Div([
            html.Div([wrap(fig1), wrap(fig2)],
                     style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),
            html.Div([
                wrap(fig3, f"Best month: {best_month}   |   Worst month: {worst_month}"),
                wrap(fig4, f"Predicted price in {pred_years} years: {pred_val:,.2f}  (Linear Regression estimate)"),
            ], style={"display": "flex", "gap": "16px"}),
        ])
        return f"{symbol} -- Analysis", status, cards, content

    # ── COMPARISON TAB ────────────────────────────────────────
    elif tab == "compare":
        if not compare_sym or not compare_sym.strip():
            return (f"{symbol}", status, cards,
                    html.Div("Enter a second stock in the sidebar under Compare With and click Run Analysis.",
                             style={"color": C["subtext"], "marginTop": "40px", "textAlign": "center"}))

        df2 = download(compare_sym.strip().upper(), start, end)
        if df2.empty:
            return (f"{symbol}", status, cards,
                    html.Div(f"No data found for {compare_sym}",
                             style={"color": C["negative"]}))

        s1  = df['Close'].dropna()
        s2  = df2['Close'].dropna()
        s1n = (s1 / safe_float(s1.iloc[0])) * 100
        s2n = (s2 / safe_float(s2.iloc[0])) * 100

        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=s1n.index, y=s1n.values.flatten(),
                                   mode='lines', name=symbol,
                                   line=dict(color=C["accent"], width=2)))
        fig_c.add_trace(go.Scatter(x=s2n.index, y=s2n.values.flatten(),
                                   mode='lines', name=compare_sym.upper(),
                                   line=dict(color=C["accent2"], width=2)))
        fig_c.update_layout(
            title=f"{symbol} vs {compare_sym.upper()} -- Normalised to 100",
            paper_bgcolor=C["card"], plot_bgcolor=C["plot_bg"],
            font_color=C["text"], margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=C["grid"]),
            yaxis=dict(gridcolor=C["grid"]),
        )

        def pct(series):
            return series.pct_change().dropna()

        s1_ret   = float(pct(s1).mean() * 100)
        s2_ret   = float(pct(s2).mean() * 100)
        s1_vol   = float(pct(s1).std() * 100)
        s2_vol   = float(pct(s2).std() * 100)
        s1_total = float(((safe_float(s1.iloc[-1]) - safe_float(s1.iloc[0])) / safe_float(s1.iloc[0])) * 100)
        s2_total = float(((safe_float(s2.iloc[-1]) - safe_float(s2.iloc[0])) / safe_float(s2.iloc[0])) * 100)

        def stat_row(label, v1, v2, higher_better=True):
            win1 = (v1 > v2) if higher_better else (v1 < v2)
            c1   = C["positive"] if win1 else C["text"]
            c2   = C["positive"] if not win1 else C["text"]
            return html.Tr([
                html.Td(label,        style={"padding": "10px 16px", "color": C["subtext"], "fontSize": "13px"}),
                html.Td(f"{v1:.2f}%", style={"padding": "10px 16px", "color": c1, "fontWeight": "600", "textAlign": "center"}),
                html.Td(f"{v2:.2f}%", style={"padding": "10px 16px", "color": c2, "fontWeight": "600", "textAlign": "center"}),
            ], style={"borderBottom": f"1px solid {C['border']}"})

        table = html.Table([
            html.Thead(html.Tr([
                html.Th("Metric",            style={"padding": "12px 16px", "textAlign": "left",   "color": C["subtext"], "fontSize": "11px", "textTransform": "uppercase"}),
                html.Th(symbol,              style={"padding": "12px 16px", "textAlign": "center", "color": C["accent"],  "fontSize": "13px"}),
                html.Th(compare_sym.upper(), style={"padding": "12px 16px", "textAlign": "center", "color": C["accent2"], "fontSize": "13px"}),
            ], style={"borderBottom": f"1px solid {C['border']}"})),
            html.Tbody([
                stat_row("Total Return",      s1_total, s2_total, True),
                stat_row("Avg Daily Return",  s1_ret,   s2_ret,   True),
                stat_row("Volatility (risk)", s1_vol,   s2_vol,   False),
            ])
        ], style={"width": "100%", "borderCollapse": "collapse",
                  "backgroundColor": C["card"], "borderRadius": "12px",
                  "border": f"1px solid {C['border']}", "marginTop": "16px"})

        content = html.Div([
            html.Div([dcc.Graph(figure=fig_c, config={"displayModeBar": False})],
                     style={"backgroundColor": C["card"], "borderRadius": "12px",
                            "border": f"1px solid {C['border']}", "padding": "8px"}),
            html.Div("Head-to-Head Stats",
                     style={"fontSize": "14px", "fontWeight": "600", "marginTop": "20px"}),
            html.Div("Green = winner in that metric",
                     style={"fontSize": "11px", "color": C["subtext"], "marginBottom": "8px"}),
            table,
        ])
        return f"{symbol} vs {compare_sym.upper()}", status, cards, content

    # ── NEWS TAB ──────────────────────────────────────────────
    elif tab == "news":
        ticker = yf.Ticker(symbol)
        news   = ticker.news
        if not news:
            return (f"{symbol} -- News", status, cards,
                    html.Div("No news found for this stock right now.",
                             style={"color": C["subtext"], "textAlign": "center", "marginTop": "40px"}))

        news_cards = []
        for item in news[:10]:
            c        = item.get("content", {})
            headline = c.get("title", "No title")
            summary  = c.get("summary", "")
            pub_date = c.get("pubDate", "")[:10]
            link     = c.get("canonicalUrl", {}).get("url", "#")
            provider = c.get("provider", {}).get("displayName", "")
            news_cards.append(html.Div([
                html.Div([
                    html.Span(provider, style={"fontSize": "10px", "color": C["accent"],
                                               "fontWeight": "600", "textTransform": "uppercase",
                                               "letterSpacing": "0.06em"}),
                    html.Span(pub_date, style={"fontSize": "10px", "color": C["subtext"],
                                               "marginLeft": "12px"}),
                ], style={"marginBottom": "6px"}),
                html.A(headline, href=link, target="_blank", style={
                    "fontSize": "14px", "fontWeight": "600", "color": C["text"],
                    "textDecoration": "none", "display": "block", "marginBottom": "6px"
                }),
                html.Div(summary[:180] + "..." if len(summary) > 180 else summary,
                         style={"fontSize": "12px", "color": C["subtext"], "lineHeight": "1.6"}),
            ], style={
                "backgroundColor": C["card"], "borderRadius": "10px",
                "border": f"1px solid {C['border']}", "padding": "16px",
                "marginBottom": "12px", "borderLeft": f"3px solid {C['accent']}"
            }))

        return (f"{symbol} -- News", status, cards,
                html.Div([
                    html.Div(f"Latest News for {symbol}",
                             style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "16px"}),
                    html.Div(news_cards)
                ]))

    return f"{symbol}", status, cards, html.Div()


if __name__ == "__main__":
    print("=" * 45)
    print("  StockIQ Dashboard starting...")
    print("  Open your browser and go to:")
    print("  http://127.0.0.1:8050")
    print("=" * 45)
    app.run(debug=False, host="0.0.0.0", port=10000)
