"""
GrihZen — signup_ui.py
Multi-step signup wizard styled to match the GrihZen brand.
"""

import re, time
import streamlit as st

# Lazy imports to avoid circular dependency with grihzen.py
def _get_helpers():
    from grihzen import create_user, create_household, authenticate_user, join_household
    return create_user, create_household, authenticate_user, join_household

# ── Brand tokens ─────────────────────────────────────────────────────────────
BRAND = {
    "ink":     "#0F2742",
    "body":    "#4A5B73",
    "muted":   "#8A96A8",
    "line":    "#E5EAF2",
    "surface": "#F6F9FD",
    "blue":    "#4A8BF0",
    "blue_d":  "#2F6FE0",
    "green":   "#2CD68E",
    "green_d": "#1FB373",
    "danger":  "#D9445F",
}
TOTAL_STEPS = 3


# ── CSS ───────────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown(f"""<style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      .stApp {{ background:#f6f9fd !important; }}
      .block-container {{ padding-top:2.5rem !important; max-width:560px; }}
      html,body,[class*="css"] {{ font-family:'Inter',system-ui,sans-serif; }}

      .gz-su-logo {{ display:flex;align-items:center;justify-content:center;gap:10px;margin-bottom:24px; }}
      .gz-su-logo-text {{ font-weight:700;font-size:22px;color:{BRAND["ink"]};letter-spacing:-.02em; }}
      .gz-su-logo-text .accent {{ color:{BRAND["blue"]}; }}

      .gz-su-card {{
        background:#fff;border:1px solid {BRAND["line"]};
        border-radius:20px;padding:32px 36px;
        box-shadow:0 24px 60px rgba(11,26,46,0.07);
      }}

      .gz-su-step-row {{ display:flex;justify-content:space-between;align-items:center;margin-bottom:18px; }}
      .gz-su-step-label {{ font-size:11px;color:{BRAND["muted"]};font-weight:700;letter-spacing:.08em;text-transform:uppercase; }}
      .gz-su-dots {{ display:flex;gap:6px;align-items:center; }}
      .gz-su-dot {{ height:6px;border-radius:3px;transition:all .2s; }}

      .gz-su-title {{ font-size:24px;font-weight:700;color:{BRAND["ink"]};letter-spacing:-.02em;margin:0 0 6px 0; }}
      .gz-su-sub {{ font-size:14px;color:{BRAND["body"]};margin:0 0 20px 0; }}

      .gz-su-grad-title {{
        background:linear-gradient(135deg,{BRAND["blue"]} 0%,{BRAND["green"]} 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;font-weight:700;
      }}

      .stTextInput>div>div>input {{
        height:44px !important;border-radius:10px !important;
        border:1.5px solid {BRAND["line"]} !important;
        font-size:15px !important;padding:0 14px !important;
        background:#fff !important;color:{BRAND["ink"]} !important;
        transition:border-color .15s,box-shadow .15s !important;
      }}
      .stTextInput>div>div>input:focus {{
        border-color:{BRAND["blue"]} !important;
        box-shadow:0 0 0 4px rgba(74,139,240,0.12) !important;
      }}
      .stTextInput label,.stRadio label,.stSelectbox label {{
        font-size:13px !important;font-weight:500 !important;color:{BRAND["ink"]} !important;
      }}

      .stButton>button[kind="primary"] {{
        background:linear-gradient(135deg,{BRAND["blue"]} 0%,{BRAND["green"]} 100%) !important;
        color:#fff !important;border:0 !important;height:46px !important;
        border-radius:10px !important;font-weight:600 !important;font-size:15px !important;
        box-shadow:0 6px 18px rgba(47,111,224,0.28) !important;
        transition:filter .15s !important;
      }}
      .stButton>button[kind="primary"]:hover {{ filter:brightness(1.05) !important; }}
      .stButton>button[kind="secondary"] {{
        background:#fff !important;color:{BRAND["body"]} !important;
        border:1.5px solid {BRAND["line"]} !important;height:46px !important;
        border-radius:10px !important;font-weight:500 !important;
      }}
      .stButton>button[kind="secondary"]:hover {{
        background:{BRAND["surface"]} !important;border-color:{BRAND["blue"]} !important;
        color:{BRAND["blue"]} !important;
      }}

      .gz-su-strength {{ display:flex;gap:4px;margin-top:6px; }}
      .gz-su-strength>div {{ flex:1;height:4px;border-radius:2px; }}
      .gz-su-hint {{ font-size:12px;color:{BRAND["muted"]};margin-top:6px; }}
      .gz-su-hint.err {{ color:{BRAND["danger"]}; }}
      .gz-su-hint.ok  {{ color:{BRAND["green_d"]}; }}

      .gz-su-invite {{
        background:linear-gradient(135deg,rgba(74,139,240,.08),rgba(44,214,142,.08));
        border-radius:12px;padding:14px 16px;margin:18px 0;
        font-family:ui-monospace,monospace;font-size:13px;color:{BRAND["ink"]};
      }}

      #MainMenu,footer,header {{ visibility:hidden; }}
    </style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _logo():
    st.markdown(f"""
    <div class="gz-su-logo">
      <svg width="32" height="32" viewBox="0 0 100 100">
        <defs>
          <linearGradient id="gzg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="{BRAND['blue']}"/>
            <stop offset="50%" stop-color="#3FB8C2"/>
            <stop offset="100%" stop-color="{BRAND['green']}"/>
          </linearGradient>
        </defs>
        <circle cx="50" cy="50" r="46" fill="none" stroke="rgba(74,139,240,0.25)" stroke-width="1.5"/>
        <line x1="22" y1="28" x2="78" y2="28" stroke="url(#gzg)" stroke-width="9" stroke-linecap="round"/>
        <line x1="78" y1="28" x2="22" y2="72" stroke="url(#gzg)" stroke-width="9" stroke-linecap="round"/>
        <line x1="22" y1="72" x2="78" y2="72" stroke="url(#gzg)" stroke-width="9" stroke-linecap="round"/>
        <circle cx="22" cy="28" r="7" fill="#fff" stroke="{BRAND['blue']}" stroke-width="2"/>
        <circle cx="78" cy="28" r="7" fill="#fff" stroke="{BRAND['blue']}" stroke-width="2"/>
        <circle cx="22" cy="72" r="7" fill="#fff" stroke="{BRAND['green']}" stroke-width="2"/>
        <circle cx="78" cy="72" r="7" fill="#fff" stroke="{BRAND['green']}" stroke-width="2"/>
      </svg>
      <span class="gz-su-logo-text">Grih<span class="accent">Zen</span></span>
    </div>""", unsafe_allow_html=True)


def _step_header(step:int, title:str, subtitle:str):
    dots=""
    for i in range(1, TOTAL_STEPS+1):
        if i < step:    color,w = BRAND["green"],"6px"
        elif i == step: color,w = BRAND["blue"],"28px"
        else:           color,w = BRAND["line"],"6px"
        dots += f'<div class="gz-su-dot" style="background:{color};width:{w}"></div>'
    st.markdown(f"""
      <div class="gz-su-step-row">
        <span class="gz-su-step-label">Step {step} of {TOTAL_STEPS}</span>
        <div class="gz-su-dots">{dots}</div>
      </div>
      <h2 class="gz-su-title">{title}</h2>
      <p class="gz-su-sub">{subtitle}</p>""", unsafe_allow_html=True)


def _password_strength(pw:str) -> int:
    if not pw: return 0
    score = 0
    if len(pw) >= 6:  score += 1
    if len(pw) >= 10: score += 1
    if re.search(r"\d", pw) and re.search(r"[A-Za-z]", pw): score += 1
    if re.search(r"[^A-Za-z0-9]", pw): score += 1
    return score

def _strength_bar(score:int):
    palette = [BRAND["line"], BRAND["danger"], "#E0A800", BRAND["green"], BRAND["green_d"]]
    label   = ["", "Too weak", "Weak", "Good", "Strong"][score]
    label_color = palette[score] if score > 0 else BRAND["muted"]
    bars = "".join(f'<div style="background:{"" if i>=score else palette[score]};{"background:"+BRAND["line"] if i>=score else ""}"></div>' for i in range(4))
    bars = "".join(f'<div style="background:{palette[score] if i<score else BRAND["line"]}"></div>' for i in range(4))
    st.markdown(f"""
      <div class="gz-su-strength">{bars}</div>
      <div class="gz-su-hint" style="color:{label_color};font-weight:600">{label}</div>""",
      unsafe_allow_html=True)


# ── Steps ─────────────────────────────────────────────────────────────────────
def _step_account():
    create_user, _, authenticate_user, _ = _get_helpers()
    _step_header(1, "Create your account",
                 "One log-in for every device. Add household members next.")

    name = st.text_input("Full name", placeholder="Asha Patel", key="su_name")
    user = st.text_input("Username", placeholder="asha_patel", key="su_user").strip().lower()
    pw1  = st.text_input("Password", type="password",
                         placeholder="At least 6 characters", key="su_pw1")
    if pw1: _strength_bar(_password_strength(pw1))
    pw2  = st.text_input("Confirm password", type="password", key="su_pw2")
    pin  = st.text_input("Recovery PIN — 4 digits, optional", max_chars=4,
                         placeholder="Used to reset your password later", key="su_pin")

    cols = st.columns([1, 1.6])
    with cols[1]:
        if st.button("Continue →", type="primary", use_container_width=True, key="su_step1_next"):
            errs=[]
            if not name.strip():  errs.append("Enter your full name.")
            if not user:          errs.append("Pick a username.")
            elif not re.match(r"^[a-z0-9_]{3,}$", user):
                errs.append("Username: 3+ chars, letters/numbers/_ only.")
            if len(pw1) < 6:      errs.append("Password must be at least 6 characters.")
            if pw1 != pw2:        errs.append("Passwords don't match.")
            if pin and not pin.isdigit(): errs.append("PIN must be digits only.")
            if errs:
                for e in errs: st.error(e)
                return
            ok = create_user(user, name.strip(), pw1, pin if pin else None)
            if not ok:
                st.error("That username is already taken — try another.")
                return
            u = authenticate_user(user, pw1)
            if u:
                st.session_state.user_id      = u[0]
                st.session_state.username     = u[1]
                st.session_state.display_name = u[2]
                st.session_state.logged_in    = True
            st.session_state.su_step = 2
            st.rerun()


def _step_household():
    _, create_household, _, join_household = _get_helpers()
    _step_header(2, "Set up your household",
                 "Like a WhatsApp group for your home — shared recipes, locations & more.")

    mode = st.radio("How do you want to start?",
                    ["Create a new household", "Join an existing one", "Skip for now"],
                    key="su_mode")

    if mode == "Create a new household":
        hname = st.text_input("Household name", placeholder="The Singh Home 🏡", key="su_hname")
        st.selectbox("How many people live here?",
                     ["Just me","2 people","3–4 people","5 or more"], index=1, key="su_hsize")
    elif mode == "Join an existing one":
        st.text_input("Invite code", placeholder="ABC123", key="su_jcode")

    cols = st.columns([1, 1.6])
    with cols[0]:
        if st.button("← Back", use_container_width=True, key="su_step2_back"):
            st.session_state.su_step = 1; st.rerun()
    with cols[1]:
        if st.button("Continue →", type="primary", use_container_width=True, key="su_step2_next"):
            uid = st.session_state.get("user_id")
            if not uid:
                st.error("Session lost — go back to step 1."); return
            if mode == "Create a new household":
                hname = st.session_state.get("su_hname","").strip()
                if not hname: st.error("Give your household a name."); return
                hid, code = create_household(hname, uid)
                st.session_state.household_id    = hid
                st.session_state.su_invite_code  = code
            elif mode == "Join an existing one":
                code = st.session_state.get("su_jcode","").strip().upper()
                if not code: st.error("Enter the invite code."); return
                hid = join_household(code, uid)
                if not hid: st.error("That invite code didn't match any household."); return
                st.session_state.household_id   = hid
                st.session_state.su_invite_code = None
            else:
                st.session_state.household_id   = -1
                st.session_state.su_invite_code = None
            st.session_state.su_step = 3
            st.rerun()


def _step_done():
    _step_header(3, "You're all set 🎉",
                 "Your dashboard is ready. Invite household members anytime from Settings.")
    display = st.session_state.get("display_name","")
    st.markdown(f"""
      <div style="text-align:center;margin:6px 0 20px;">
        <div style="display:inline-flex;align-items:center;justify-content:center;
                    width:72px;height:72px;border-radius:50%;
                    background:linear-gradient(135deg,{BRAND['blue']},{BRAND['green']});
                    box-shadow:0 18px 40px rgba(44,214,142,0.35);">
          <svg width="34" height="34" viewBox="0 0 24 24" fill="none">
            <path d="M4 12l5 5L20 6" stroke="#fff" stroke-width="2.6"
                  stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <h2 style="margin:16px 0 4px;font-size:24px;color:{BRAND['ink']};">
          Welcome to <span class="gz-su-grad-title">GrihZen</span>, {display}!
        </h2>
        <p style="color:{BRAND['body']};font-size:14px;">Your household hub is ready.</p>
      </div>""", unsafe_allow_html=True)

    code = st.session_state.get("su_invite_code")
    if code:
        st.markdown(f"""
          <div class="gz-su-invite">
            <div style="font-weight:600;color:{BRAND['ink']};margin-bottom:4px;">🔗 Household Invite Code</div>
            <div style="font-size:20px;letter-spacing:.1em;font-weight:700;">{code}</div>
            <div style="font-size:12px;color:{BRAND['body']};margin-top:6px;">
              Share this with family members so they can join.
            </div>
          </div>""", unsafe_allow_html=True)

    if st.button("Go to dashboard →", type="primary", use_container_width=True, key="su_finish"):
        for k in ("su_step","su_name","su_user","su_pw1","su_pw2","su_pin",
                  "su_mode","su_hname","su_hsize","su_jcode","su_invite_code","su_show"):
            st.session_state.pop(k, None)
        st.rerun()


# ── Public entry point ────────────────────────────────────────────────────────
def show_signup_wizard():
    _inject_css()
    _logo()
    st.markdown('<div class="gz-su-card">', unsafe_allow_html=True)
    step = st.session_state.get("su_step", 1)
    if step == 1:   _step_account()
    elif step == 2: _step_household()
    else:           _step_done()
    st.markdown('</div>', unsafe_allow_html=True)

    if step == 1:
        st.markdown(
            f'<div style="text-align:center;margin-top:18px;font-size:13px;color:{BRAND["body"]};">'
            f'Already have an account? </div>', unsafe_allow_html=True)
        if st.button("← Sign in instead", use_container_width=True, key="su_back_to_login"):
            st.session_state.su_show = False
            st.session_state.pop("su_step", None)
            st.rerun()
