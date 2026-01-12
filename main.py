import base64
import hashlib
import hmac
import json
import math
import os
import secrets
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from psycopg2.extras import RealDictCursor

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR.parent / ".env")
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

MOVIE_DATABASE_URL = os.getenv("MOVIE_DATABASE_URL")
USER_DATABASE_URL = os.getenv("USER_DATABASE_URL")
FAVORITES_DATABASE_URL = os.getenv("FAVORITES_DATABASE_URL")
IS_PREMIUM_DATABASE_URL = os.getenv("IS_PREMIUM_DATABASE_URL")
TIER_ONE_STRIPE_ID = os.getenv("TIER_ONE_STRIPE_ID", "")
PREMIUM_PRICE_TEXT = os.getenv("PREMIUM_PRICE_TEXT", "$4.99/month")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_API_KEY = os.getenv("LLM_API_KEY")
CANONICAL_PATH = os.getenv("CANONICAL_PATH", str(BASE_DIR / "canonical_top_k1000.txt"))
IMG_BASE = "https://image.tmdb.org/t/p/w342"
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", "604800"))
FREE_FAVORITES_LIMIT = 10
ACTIVE_SUBSCRIPTION_STATUSES = {"active", "trialing"}

LANG_NAMES = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "bg": "Bulgarian",
    "bm": "Bambara",
    "bn": "Bengali",
    "bo": "Tibetan",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cn": "Cantonese",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dz": "Dzongkha",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "iu": "Inuktitut",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ks": "Kashmiri",
    "ku": "Kurdish",
    "ky": "Kyrgyz",
    "ln": "Lingala",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mo": "Moldovan",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "nb": "Norwegian Bokmal",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "or": "Odia",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "se": "Northern Sami",
    "sh": "Serbo-Croatian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tn": "Tswana",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wo": "Wolof",
    "xx": "Unknown",
    "yi": "Yiddish",
    "zh": "Chinese",
    "zu": "Zulu",
}

COMMON_LANG_ORDER = [
    "en",
    "zh",
    "cn",
    "es",
    "hi",
    "ar",
    "pt",
    "ru",
    "ja",
    "de",
    "fr",
    "ko",
    "it",
]

SAMPLE_LANGS = ["en", "es", "fr", "ja"]
SAMPLE_FEATURED = [
    {
        "title": "Avatar: Fire and Ash",
        "release_date": "2025-12-17",
        "rating": "PG-13",
        "runtime": 198,
        "popularity": 490.0904,
        "genres": [
            "Science Fiction",
            "Adventure",
            "Fantasy",
        ],
        "poster_path": "/g96wHxU7EnoIFwemb2RgohIXrgW.jpg",
        "keywords_topics": [
            "aliens",
            "interstellar war",
            "indigenous culture",
            "military dictatorship",
            "power struggle",
            "family",
        ],
        "overview": "In the wake of the devastating war against the RDA and the loss of their eldest son, Jake Sully and Neytiri face a new threat on Pandora: the Ash People, a violent and power-hungry Na'vi tribe led by the ruthless Varang. Jake's family must fight for their survival and the future of Pandora in a conflict that pushes them to their emotional and physical limits.",
    },
    {
        "title": "B\u0101hubali: The Epic",
        "release_date": "2025-10-29",
        "rating": "NR",
        "runtime": 224,
        "popularity": 440.7241,
        "genres": [
            "Action",
            "Drama",
        ],
        "poster_path": "/4sLSorDKKDN944kWngxgQlpdDeg.jpg",
        "keywords_topics": [
            "high fantasy",
            "medieval",
            "monarchy",
            "whodunit",
            "sibling rivalry",
            "revenge",
        ],
        "overview": "When a mysterious child is found by a tribal couple near a roaring waterfall, they raise him as their own. As he grows, Sivudu is drawn to the world beyond the cliffs, where he discovers the ancient kingdom of Mahishmati, ruled by a cruel tyrant, haunted by rebellion, and bound to his past. What begins as a quest for love soon unravels a legacy of betrayal, sacrifice, and a forgotten prince.",
    },
    {
        "title": "Zootopia 2",
        "release_date": "2025-11-26",
        "rating": "PG",
        "runtime": 107,
        "popularity": 380.0296,
        "genres": [
            "Animation",
            "Comedy",
            "Adventure",
            "Family",
            "Mystery",
        ],
        "poster_path": "/bjUWGw0Ao0qVWxagN3VCwBJHVo6.jpg",
        "keywords_topics": [
            "animation",
            "buddy cop",
            "animals/talking animals",
            "crime",
            "illustration",
        ],
        "overview": "After cracking the biggest case in Zootopia's history, rookie cops Judy Hopps and Nick Wilde find themselves on the twisting trail of a great mystery when Gary De'Snake arrives and turns the animal metropolis upside down. To crack the case, Judy and Nick must go undercover to unexpected new parts of town, where their growing partnership is tested like never before.",
    },
    {
        "title": "Demon Slayer: Kimetsu no Yaiba Infinity Castle",
        "release_date": "2025-07-18",
        "rating": "R",
        "runtime": 156,
        "popularity": 221.0241,
        "genres": [
            "Animation",
            "Action",
            "Fantasy",
        ],
        "poster_path": "/fWVSwgjpT2D78VUh6X8UBd2rorW.jpg",
        "keywords_topics": [
            "demons",
            "dark fantasy",
            "supernatural",
            "action",
            "battle",
            "animation",
        ],
        "overview": "The Demon Slayer Corps are drawn into the Infinity Castle, where Tanjiro, Nezuko, and the Hashira face terrifying Upper Rank demons in a desperate fight as the final battle against Muzan Kibutsuji begins.",
    },
    {
        "title": "Avatar: The Way of Water",
        "release_date": "2022-12-14",
        "rating": "PG-13",
        "runtime": 192,
        "popularity": 123.8467,
        "genres": [
            "Action",
            "Adventure",
            "Science Fiction",
        ],
        "poster_path": "/t6HIqrRAclMCA60NsSmeqe9RmNV.jpg",
        "keywords_topics": [
            "aliens",
            "interstellar war",
            "underwater",
            "native americans",
            "family",
            "colonialism",
        ],
        "overview": "Set more than a decade after the events of the first film, learn the story of the Sully family (Jake, Neytiri, and their kids), the trouble that follows them, the lengths they go to keep each other safe, the battles they fight to stay alive, and the tragedies they endure.",
    },
    {
        "title": "The Housemaid",
        "release_date": "2025-12-18",
        "rating": "R",
        "runtime": 131,
        "popularity": 123.4737,
        "genres": [
            "Mystery",
            "Thriller",
        ],
        "poster_path": "/cWsBscZzwu5brg9YjNkGewRUvJX.jpg",
        "keywords_topics": [
            "psychological thriller",
            "domestic violence",
            "servants",
            "family legacy",
            "conspiracy theories",
            "trauma",
        ],
        "overview": "Trying to escape her past, Millie Calloway accepts a job as a live-in housemaid for the wealthy Nina and Andrew Winchester. But what begins as a dream job quickly unravels into something far more dangerous\u2014a sexy, seductive game of secrets, scandal, and power.",
    },
    {
        "title": "The Shadow's Edge",
        "release_date": "2025-08-16",
        "rating": "NR",
        "runtime": 142,
        "popularity": 123.058,
        "genres": [
            "Action",
            "Crime",
            "Thriller",
        ],
        "poster_path": "/e0RU6KpdnrqFxDKlI3NOqN8nHL6.jpg",
        "keywords_topics": [
            "heist",
            "surveillance",
            "crime",
            "investigation",
            "police",
        ],
        "overview": "Macau Police brings the tracking expert police officer out of retirement to help catch a dangerous group of professional thieves.",
    },
    {
        "title": "Predator: Badlands",
        "release_date": "2025-11-05",
        "rating": "PG-13",
        "runtime": 107,
        "popularity": 122.7678,
        "genres": [
            "Action",
            "Science Fiction",
            "Adventure",
        ],
        "poster_path": "/ef2QSeBkrYhAdfsWGXmp0lvH0T1.jpg",
        "keywords_topics": [
            "aliens",
            "space",
            "dragons",
            "survival",
            "agriculture",
            "coming-of-age",
            "corporate corruption",
        ],
        "overview": "Cast out from his clan, a young Predator finds an unlikely ally in a damaged android and embarks on a treacherous journey in search of the ultimate adversary.",
    },
]

SAMPLE_RESULTS = [
    {
        "title": "Iron Skies",
        "release_date": "2019-06-12",
        "rating": "7.1",
        "runtime": 112,
        "popularity": 6.8,
        "genres": ["Action", "Sci-Fi"],
        "poster_path": None,
        "keywords_topics": ["dystopian", "war", "resistance"],
        "match_count": 3,
        "overview": "A rebel pilot leads a final strike against the regime.",
    },
    {
        "title": "Clockwork Wars",
        "release_date": "2020-09-02",
        "rating": "6.8",
        "runtime": 109,
        "popularity": 6.5,
        "genres": ["Drama", "Sci-Fi"],
        "poster_path": None,
        "keywords_topics": ["robot", "future", "loss"],
        "match_count": 2,
        "overview": "A grieving inventor confronts the sentient machine he built.",
    },
    {
        "title": "Quiet Orbit",
        "release_date": "2021-01-25",
        "rating": "6.6",
        "runtime": 101,
        "popularity": 6.2,
        "genres": ["Sci-Fi", "Thriller"],
        "poster_path": None,
        "keywords_topics": ["survival", "isolation", "space"],
        "match_count": 1,
        "overview": "A lone astronaut fights to keep a drifting station alive.",
    },
]


def format_language(code: str) -> str:
    return LANG_NAMES.get(code, code)


def order_languages(languages: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for code in COMMON_LANG_ORDER:
        if code in languages and code not in seen:
            ordered.append(code)
            seen.add(code)
    remaining = [code for code in languages if code not in seen]
    remaining.sort(key=lambda code: format_language(code).lower())
    ordered.extend(remaining)
    return ordered


def build_lang_options(languages: List[str]) -> List[Dict[str, str]]:
    options = [{"value": "Any", "label": "Any"}]
    for code in order_languages(languages):
        options.append({"value": code, "label": format_language(code)})
    return options


def get_conn():
    if not MOVIE_DATABASE_URL:
        raise RuntimeError("Movie database is not configured.")
    return psycopg2.connect(MOVIE_DATABASE_URL, sslmode="require")


def get_user_conn():
    if not USER_DATABASE_URL:
        raise RuntimeError("User database is not configured.")
    return psycopg2.connect(USER_DATABASE_URL, sslmode="require")


def get_premium_conn():
    if not IS_PREMIUM_DATABASE_URL:
        raise RuntimeError("Premium database is not configured.")
    return psycopg2.connect(IS_PREMIUM_DATABASE_URL, sslmode="require")


def get_favorites_conn():
    if not FAVORITES_DATABASE_URL:
        raise RuntimeError("Favorites database is not configured.")
    return psycopg2.connect(FAVORITES_DATABASE_URL, sslmode="require")


def normalize_email(email: str) -> str:
    return email.strip().lower()


def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def base64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    iterations = 200_000
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64url_encode(salt)}${base64url_encode(derived)}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        alg, iter_str, salt_b64, hash_b64 = stored_hash.split("$", 3)
        if alg != "pbkdf2_sha256":
            return False
        iterations = int(iter_str)
        salt = base64url_decode(salt_b64)
        expected = base64url_decode(hash_b64)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(derived, expected)
    except Exception:
        return False


def create_token(email: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_TTL_SECONDS,
    }
    header_b64 = base64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_b64 = base64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{base64url_encode(signature)}"


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".", 2)
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        signature = base64url_decode(signature_b64)
        expected = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(base64url_decode(payload_b64).decode("utf-8"))
        exp = payload.get("exp")
        if exp and int(exp) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def get_current_user(request: Request) -> Optional[str]:
    if not JWT_SECRET:
        return None
    token = request.cookies.get("session", "")
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    return payload.get("sub")


def issue_session_cookie(response: Response, email: str, request: Request) -> None:
    token = create_token(email)
    response.set_cookie(
        "session",
        token,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=JWT_TTL_SECONDS,
        path="/",
    )


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_user_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, email, password_hash FROM app_users WHERE email = %s;", (email,))
        return cur.fetchone()


def create_user(email: str, password: str) -> bool:
    password_hash = hash_password(password)
    try:
        with get_user_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO app_users (email, password_hash) VALUES (%s, %s);",
                (email, password_hash),
            )
        return True
    except psycopg2.IntegrityError:
        return False


def user_email_exists(email: str) -> bool:
    with get_user_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM app_users WHERE email = %s LIMIT 1;", (email,))
        return cur.fetchone() is not None


def update_user_email(old_email: str, new_email: str) -> bool:
    try:
        with get_user_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE app_users SET email = %s WHERE email = %s;", (new_email, old_email))
            return cur.rowcount > 0
    except psycopg2.IntegrityError:
        return False


def update_user_password(email: str, new_password: str) -> bool:
    password_hash = hash_password(new_password)
    with get_user_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE app_users SET password_hash = %s WHERE email = %s;",
            (password_hash, email),
        )
        return cur.rowcount > 0

def get_premium_record(email: str) -> Optional[Dict[str, Any]]:
    if not IS_PREMIUM_DATABASE_URL:
        return None
    try:
        with get_premium_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT is_premium, stripe_subscription_id, cancel_at_period_end, current_period_end
                FROM premium_users
                WHERE user_email = %s;
                """,
                (email,),
            )
            return cur.fetchone()
    except Exception:
        return None


def is_premium_active(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return False
    if record.get("is_premium"):
        return True
    if record.get("cancel_at_period_end") and record.get("current_period_end"):
        try:
            return record["current_period_end"] > datetime.now(timezone.utc)
        except Exception:
            return False
    return False


def get_user_is_premium(email: str) -> bool:
    record = get_premium_record(email)
    return is_premium_active(record)


def set_user_premium(
    email: str,
    is_premium: bool,
    subscription_id: Optional[str] = None,
    period_end: Optional[datetime] = None,
    cancel_at_period_end: Optional[bool] = None,
    clear_subscription: bool = False,
) -> None:
    if not IS_PREMIUM_DATABASE_URL:
        return
    try:
        with get_premium_conn() as conn, conn.cursor() as cur:
            if clear_subscription:
                cur.execute(
                    """
                    INSERT INTO premium_users (user_email, is_premium, stripe_subscription_id, cancel_at_period_end, current_period_end)
                    VALUES (%s, %s, NULL, FALSE, NULL)
                    ON CONFLICT (user_email) DO UPDATE SET
                        is_premium = EXCLUDED.is_premium,
                        stripe_subscription_id = NULL,
                        cancel_at_period_end = FALSE,
                        current_period_end = NULL;
                    """,
                    (email, is_premium),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO premium_users (user_email, is_premium, stripe_subscription_id, cancel_at_period_end, current_period_end)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_email) DO UPDATE SET
                        is_premium = EXCLUDED.is_premium,
                        stripe_subscription_id = COALESCE(EXCLUDED.stripe_subscription_id, premium_users.stripe_subscription_id),
                        cancel_at_period_end = COALESCE(EXCLUDED.cancel_at_period_end, premium_users.cancel_at_period_end),
                        current_period_end = COALESCE(EXCLUDED.current_period_end, premium_users.current_period_end);
                    """,
                    (email, is_premium, subscription_id, cancel_at_period_end, period_end),
                )
    except Exception:
        return


def update_premium_by_subscription_id(
    subscription_id: str,
    is_premium: bool,
    period_end: Optional[datetime] = None,
    cancel_at_period_end: Optional[bool] = None,
    clear_subscription: bool = False,
) -> None:
    if not IS_PREMIUM_DATABASE_URL:
        return
    try:
        with get_premium_conn() as conn, conn.cursor() as cur:
            if clear_subscription:
                cur.execute(
                    """
                    UPDATE premium_users
                    SET is_premium = %s,
                        stripe_subscription_id = NULL,
                        cancel_at_period_end = FALSE,
                        current_period_end = NULL
                    WHERE stripe_subscription_id = %s;
                    """,
                    (is_premium, subscription_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE premium_users
                    SET is_premium = %s,
                        cancel_at_period_end = COALESCE(%s, cancel_at_period_end),
                        current_period_end = COALESCE(%s, current_period_end)
                    WHERE stripe_subscription_id = %s;
                    """,
                    (is_premium, cancel_at_period_end, period_end, subscription_id),
                )
    except Exception:
        return


def build_favorite_key(title: str, release_date: str) -> str:
    safe_title = (title or "").strip()
    safe_date = (release_date or "").strip()
    return f"{safe_title}|{safe_date}"


def get_favorites_for_user(email: str) -> List[Dict[str, Any]]:
    with get_favorites_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT movie_title, release_date, poster_path, genres
            FROM favorites
            WHERE user_email = %s
            ORDER BY created_at DESC;
            """,
            (email,),
        )
        return cur.fetchall()


def update_favorites_email(old_email: str, new_email: str) -> None:
    if not FAVORITES_DATABASE_URL:
        return
    with get_favorites_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE favorites SET user_email = %s WHERE user_email = %s;",
            (new_email, old_email),
        )


def update_premium_email(old_email: str, new_email: str) -> None:
    if not IS_PREMIUM_DATABASE_URL:
        return
    with get_premium_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE premium_users SET user_email = %s WHERE user_email = %s;",
            (new_email, old_email),
        )


def favorite_exists_for_user(email: str, title: str, release_date: str) -> bool:
    release_date = release_date or ""
    with get_favorites_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM favorites
            WHERE user_email = %s AND movie_title = %s AND release_date = %s
            LIMIT 1;
            """,
            (email, title, release_date),
        )
        return cur.fetchone() is not None


def count_favorites_for_user(email: str) -> int:
    with get_favorites_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM favorites WHERE user_email = %s;", (email,))
        row = cur.fetchone()
        return int(row[0]) if row else 0


def remove_favorite_for_user(email: str, title: str, release_date: str) -> None:
    release_date = release_date or ""
    with get_favorites_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM favorites
            WHERE user_email = %s AND movie_title = %s AND release_date = %s;
            """,
            (email, title, release_date),
        )


def add_favorite_for_user(
    email: str,
    title: str,
    release_date: str,
    poster_path: str,
    genres: str,
) -> None:
    release_date = release_date or ""
    with get_favorites_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO favorites (user_email, movie_title, release_date, poster_path, genres)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (email, title, release_date, poster_path or "", genres or ""),
        )

@lru_cache(maxsize=1)
def get_embedder():
    if not LLM_API_KEY:
        raise RuntimeError("Embeddings are not configured.")
    return OpenAI(api_key=LLM_API_KEY)


@lru_cache(maxsize=1)
def load_canonical_tags() -> List[str]:
    tags: List[str] = []
    try:
        with open(CANONICAL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if "\t" in line:
                    tag = line.split("\t", 1)[0].strip()
                else:
                    tag = line.strip()
                if tag:
                    tags.append(tag)
    except FileNotFoundError as exc:
        raise RuntimeError("Canonical tag file not found.") from exc
    return tags


@lru_cache(maxsize=1)
def embed_canonicals():
    tags = load_canonical_tags()
    client = get_embedder()
    resp = client.embeddings.create(model=EMBED_MODEL, input=tags)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return tags, vectors


def map_query_to_tags(query: str) -> List[str]:
    tokens = [t.strip() for t in query.split(",") if t.strip()]
    if not tokens:
        return []
    if not LLM_API_KEY:
        return tokens
    try:
        tags, canon_vectors = embed_canonicals()
        client = get_embedder()
        resp = client.embeddings.create(model=EMBED_MODEL, input=tokens)
        token_vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(canon_vectors, axis=1) + 1e-8

        chosen: List[str] = []
        seen = set()
        for vec in token_vectors:
            qnorm = np.linalg.norm(vec) + 1e-8
            sims = (canon_vectors @ vec) / (norms * qnorm)
            idx = int(sims.argmax())
            tag = tags[idx]
            if tag not in seen:
                seen.add(tag)
                chosen.append(tag)
        return chosen
    except Exception:
        return tokens


def load_languages() -> List[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT language FROM movies WHERE language IS NOT NULL ORDER BY language;")
        return [row[0] for row in cur.fetchall()]


def fetch_movies(language: Optional[str], tags: List[str], limit: int = 50) -> List[Dict[str, Any]]:
    clauses = []
    params: List[Any] = []

    has_tags = bool(tags)

    match_expr = "0"
    if has_tags:
        match_expr = (
            "cardinality((SELECT array(SELECT unnest(keywords_topics_canonical) INTERSECT "
            "SELECT unnest(%s::text[]))))"
        )
        params.append(tags)

    if language and language != "Any":
        clauses.append("language = %s")
        params.append(language)

    if has_tags:
        clauses.append("keywords_topics_canonical && %s::text[]")
        params.append(tags)

    where = "WHERE " + " AND ".join(clauses) if clauses else ""

    sql = f"""
        SELECT title, release_date, rating, runtime, popularity, genres, poster_path, overview,
               keywords_topics_canonical AS keywords_topics,
               {match_expr} AS match_count
        FROM movies
        {where}
        ORDER BY match_count DESC, popularity DESC NULLS LAST
        LIMIT %s;
    """
    params.append(limit)

    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fetch_featured(limit: int = 8) -> List[Dict[str, Any]]:
    sql = """
        SELECT title, release_date, rating, runtime, popularity, genres, poster_path, overview,
               keywords_topics_canonical AS keywords_topics
        FROM movies
        ORDER BY popularity DESC NULLS LAST, release_date DESC NULLS LAST
        LIMIT %s;
    """
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        return cur.fetchall()


def match_label(match_count: int, tag_count: int) -> str:
    if tag_count <= 0:
        return ""
    if match_count <= 0:
        return "No Match"
    high = max(1, math.floor(0.8 * tag_count))
    moderate = max(1, math.floor(0.6 * tag_count))
    low = max(1, math.floor(0.4 * tag_count))
    very_low = max(1, math.floor(0.2 * tag_count))
    if match_count >= tag_count:
        return "Very High Match"
    if match_count >= high:
        return "High Match"
    if match_count >= moderate:
        return "Moderate Match"
    if match_count >= low:
        return "Low Match"
    if match_count >= very_low:
        return "Very Low Match"
    return "No Match"


def prepare_card(
    row: Dict[str, Any],
    chosen_tags: List[str],
    pill_limit: int,
    favorite_keys: Optional[set] = None,
) -> Dict[str, Any]:
    title_text = row.get("title", "") or ""
    rating_value = row.get("rating")
    runtime_value = row.get("runtime")
    rating = str(rating_value) if rating_value else ""
    runtime = str(runtime_value) if runtime_value else ""
    overview = row.get("overview", "") or ""
    meta_parts = []
    if row.get("release_date"):
        meta_parts.append(str(row["release_date"]))
    if rating:
        meta_parts.append(rating)
    if runtime:
        meta_parts.append(f"{runtime} min")
    meta = " - ".join(meta_parts)
    genres = ", ".join(row.get("genres") or [])
    keywords = row.get("keywords_topics") or []
    matched = ""
    if chosen_tags:
        matched_set = sorted(set(keywords) & set(chosen_tags))
        matched = ", ".join(matched_set)
    poster_path = row.get("poster_path")
    poster_url = f"{IMG_BASE}{poster_path}" if poster_path else None
    release_date = str(row.get("release_date") or "")
    favorite_key = build_favorite_key(title_text, release_date)
    is_favorite = favorite_keys is not None and favorite_key in favorite_keys
    return {
        "title": title_text,
        "meta": meta,
        "rating": rating,
        "runtime": runtime,
        "genres": genres,
        "poster_url": poster_url,
        "poster_path": poster_path or "",
        "release_date": release_date,
        "favorite_key": favorite_key,
        "is_favorite": is_favorite,
        "overview": overview,
        "pills": keywords[:pill_limit],
        "matched": matched,
    }


def group_results(
    results: List[Dict[str, Any]],
    chosen_tags: List[str],
    favorite_keys: Optional[set] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    tag_count = len(chosen_tags)
    for row in results:
        label = match_label(int(row.get("match_count") or 0), tag_count)
        grouped.setdefault(label, []).append(
            prepare_card(row, chosen_tags, pill_limit=10, favorite_keys=favorite_keys)
        )

    label_order = [
        "Very High Match",
        "High Match",
        "Moderate Match",
        "Low Match",
        "Very Low Match",
        "No Match",
    ]
    grouped_list = []
    for label in label_order:
        items = grouped.get(label, [])
        if items:
            grouped_list.append({"label": label, "items": items, "count": len(items)})
    return grouped_list


def render_page(
    request: Request,
    overview_query: str,
    language_value: Optional[str],
    has_search: bool,
) -> HTMLResponse:
    current_user = get_current_user(request)
    is_premium = False
    notice = None
    use_sample = False
    favorite_keys: set = set()
    favorites_enabled = bool(FAVORITES_DATABASE_URL)

    if current_user and favorites_enabled:
        try:
            favorites_rows = get_favorites_for_user(current_user)
            favorite_keys = {
                build_favorite_key(row.get("movie_title", ""), row.get("release_date", ""))
                for row in favorites_rows
            }
        except Exception:
            favorites_enabled = False
            if not notice:
                notice = "Favorites are unavailable right now."

    if current_user:
        is_premium = get_user_is_premium(current_user)

    if not MOVIE_DATABASE_URL:
        use_sample = True
        notice = "Search is unavailable right now. Showing sample content."
        languages = SAMPLE_LANGS
    else:
        try:
            languages = load_languages()
        except Exception as exc:
            use_sample = True
            notice = "Search is unavailable right now. Showing sample content."
            languages = SAMPLE_LANGS

    lang_options = build_lang_options(languages)
    default_language = "en" if "en" in languages else (languages[0] if languages else "Any")

    if not language_value:
        selected_language = default_language
    else:
        selected_language = language_value
        if not any(opt["value"] == selected_language for opt in lang_options):
            selected_language = default_language

    query_language = None if selected_language == "Any" else selected_language

    chosen_tags: List[str] = []
    featured: List[Dict[str, Any]] = []
    grouped_results: List[Dict[str, Any]] = []
    results_count = 0
    results_limit = 50 if is_premium else 10

    if has_search:
        if overview_query:
            chosen_tags = map_query_to_tags(overview_query)
        elif use_sample:
            chosen_tags = ["dystopian", "robot", "war"]

        if use_sample:
            results = SAMPLE_RESULTS[:results_limit]
        else:
            try:
                results = fetch_movies(query_language, chosen_tags, limit=results_limit)
            except Exception as exc:
                notice = "Search is unavailable right now. Showing sample content."
                results = SAMPLE_RESULTS[:results_limit]

        grouped_results = group_results(results, chosen_tags, favorite_keys=favorite_keys)
        results_count = len(results)
    else:
        featured_rows = SAMPLE_FEATURED

        featured = [
            prepare_card(row, [], pill_limit=5, favorite_keys=favorite_keys) for row in featured_rows
        ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "overview_query": overview_query,
            "lang_options": lang_options,
            "selected_language": selected_language,
            "has_search": has_search,
            "featured": featured,
            "grouped_results": grouped_results,
            "results_count": results_count,
            "chosen_tags": chosen_tags,
            "notice": notice,
            "current_user": current_user,
            "favorites_enabled": favorites_enabled,
            "is_premium": is_premium,
        },
    )


def render_login(request: Request, message: str = "", error: str = "") -> HTMLResponse:
    current_user = get_current_user(request)
    is_premium = get_user_is_premium(current_user) if current_user else False
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "message": message,
            "error": error,
            "current_user": current_user,
            "is_premium": is_premium,
        },
    )


def favorite_row_to_card(row: Dict[str, Any], favorite_keys: set) -> Dict[str, Any]:
    title = row.get("movie_title", "") or ""
    release_date = str(row.get("release_date") or "")
    genres = row.get("genres") or ""
    poster_path = row.get("poster_path") or ""
    poster_url = f"{IMG_BASE}{poster_path}" if poster_path else None
    favorite_key = build_favorite_key(title, release_date)
    return {
        "title": title,
        "meta": release_date,
        "rating": "",
        "runtime": "",
        "genres": genres,
        "poster_url": poster_url,
        "poster_path": poster_path,
        "release_date": release_date,
        "favorite_key": favorite_key,
        "is_favorite": favorite_key in favorite_keys,
        "overview": "",
        "pills": [],
        "matched": "",
    }


def render_favorites_page(
    request: Request,
    favorites: List[Dict[str, Any]],
    error: str = "",
) -> HTMLResponse:
    current_user = get_current_user(request)
    is_premium = get_user_is_premium(current_user) if current_user else False
    return templates.TemplateResponse(
        "favorites.html",
        {
            "request": request,
            "favorites": favorites,
            "error": error,
            "current_user": current_user,
            "favorites_enabled": bool(FAVORITES_DATABASE_URL),
            "is_premium": is_premium,
        },
    )


def render_settings(
    request: Request,
    email_message: str = "",
    email_error: str = "",
    password_message: str = "",
    password_error: str = "",
    subscription_message: str = "",
    subscription_error: str = "",
    current_user_override: Optional[str] = None,
) -> HTMLResponse:
    current_user = current_user_override or get_current_user(request)
    premium_record = get_premium_record(current_user) if current_user else None
    is_premium = is_premium_active(premium_record)
    subscription_id = premium_record.get("stripe_subscription_id") if premium_record else None
    cancel_at_period_end = bool(premium_record.get("cancel_at_period_end")) if premium_record else False
    current_period_end = premium_record.get("current_period_end") if premium_record else None
    period_end_text = ""
    if current_period_end:
        try:
            period_end_text = current_period_end.astimezone(timezone.utc).strftime("%b %d, %Y")
        except Exception:
            period_end_text = str(current_period_end)
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "current_user": current_user,
            "is_premium": is_premium,
            "subscription_id": subscription_id,
            "cancel_at_period_end": cancel_at_period_end,
            "period_end_text": period_end_text,
            "email_message": email_message,
            "email_error": email_error,
            "password_message": password_message,
            "password_error": password_error,
            "subscription_message": subscription_message,
            "subscription_error": subscription_error,
        },
    )


def render_premium_page(request: Request, message: str = "", error: str = "") -> HTMLResponse:
    current_user = get_current_user(request)
    is_premium = get_user_is_premium(current_user) if current_user else False
    premium_record = get_premium_record(current_user) if current_user else None
    cancel_at_period_end = premium_record.get("cancel_at_period_end") if premium_record else False
    current_period_end = premium_record.get("current_period_end") if premium_record else None
    period_end_text = ""
    if current_period_end:
        try:
            period_end_text = current_period_end.astimezone(timezone.utc).strftime("%b %d, %Y")
        except Exception:
            period_end_text = str(current_period_end)
    return templates.TemplateResponse(
        "premium.html",
        {
            "request": request,
            "current_user": current_user,
            "is_premium": is_premium,
            "cancel_at_period_end": cancel_at_period_end,
            "period_end_text": period_end_text,
            "premium_price_text": PREMIUM_PRICE_TEXT,
            "stripe_publishable_key": STRIPE_PUBLISHABLE_KEY,
            "message": message,
            "error": error,
        },
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_page(request, overview_query="", language_value=None, has_search=False)


@app.post("/", response_class=HTMLResponse)
def search(
    request: Request,
    overview_query: str = Form(""),
    language: str = Form("Any"),
):
    return render_page(request, overview_query=overview_query, language_value=language, has_search=True)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return render_login(request)


@app.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    email: str = Form(""),
    password: str = Form(""),
):
    if not USER_DATABASE_URL:
        return render_login(request, error="Login is unavailable right now.")
    if not JWT_SECRET:
        return render_login(request, error="Login is unavailable right now.")
    if not email or not password:
        return render_login(request, error="Please enter your email and password.")
    email_clean = normalize_email(email)
    try:
        user = get_user_by_email(email_clean)
    except Exception:
        return render_login(request, error="Login is unavailable. Please try again shortly.")
    if not user:
        return render_login(request, error="Invalid email or incorrect password.")
    if not verify_password(password, user.get("password_hash", "")):
        return render_login(request, error="Invalid email or incorrect password.")
    response = RedirectResponse(url="/", status_code=303)
    issue_session_cookie(response, email_clean, request)
    return response


@app.post("/register", response_class=HTMLResponse)
def register_submit(
    request: Request,
    email: str = Form(""),
    password: str = Form(""),
    confirm_password: str = Form(""),
):
    if not USER_DATABASE_URL:
        return render_login(request, error="Registration is unavailable right now.")
    if not JWT_SECRET:
        return render_login(request, error="Registration is unavailable right now.")
    if not email or not password:
        return render_login(request, error="Please fill in email and password.")
    if password != confirm_password:
        return render_login(request, error="Passwords do not match.")
    email_clean = normalize_email(email)
    try:
        created = create_user(email_clean, password)
    except Exception:
        return render_login(request, error="Registration is unavailable. Please try again shortly.")
    if not created:
        return render_login(request, error="Registration failed. Please try again.")
    response = render_login(request, message="Account created.")
    issue_session_cookie(response, email_clean, request)
    return response


@app.get("/favorites", response_class=HTMLResponse)
def favorites_page(request: Request):
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    if not FAVORITES_DATABASE_URL:
        return render_favorites_page(request, [], error="Favorites are unavailable right now.")
    try:
        rows = get_favorites_for_user(current_user)
        favorite_keys = {
            build_favorite_key(row.get("movie_title", ""), row.get("release_date", ""))
            for row in rows
        }
        cards = [favorite_row_to_card(row, favorite_keys) for row in rows]
    except Exception:
        return render_favorites_page(request, [], error="Favorites are unavailable right now.")
    return render_favorites_page(request, cards)


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    return render_settings(request)


@app.post("/settings/email", response_class=HTMLResponse)
def settings_change_email(
    request: Request,
    new_email: str = Form(""),
    current_password: str = Form(""),
):
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    email_clean = normalize_email(new_email)
    if not email_clean:
        return render_settings(request, email_error="Enter a valid email.")
    if email_clean == current_user:
        return render_settings(request, email_error="That is already your email.")
    if user_email_exists(email_clean):
        return render_settings(request, email_error="That email is already in use.")

    user = get_user_by_email(current_user)
    if not user or not verify_password(current_password, user.get("password_hash", "")):
        return render_settings(request, email_error="Current password is incorrect.")

    premium_conflict = get_premium_record(email_clean)
    if premium_conflict:
        return render_settings(request, email_error="That email already has a subscription record.")

    updated = update_user_email(current_user, email_clean)
    if not updated:
        return render_settings(request, email_error="Could not update email. Try again.")

    warning = ""
    try:
        update_favorites_email(current_user, email_clean)
    except Exception:
        warning = "Email updated, but favorites could not be updated."
    try:
        update_premium_email(current_user, email_clean)
    except Exception:
        warning = "Email updated, but subscription data could not be updated."

    message = "Email updated."
    if warning:
        message = f"{message} {warning}"
    response = render_settings(request, email_message=message, current_user_override=email_clean)
    issue_session_cookie(response, email_clean, request)
    return response


@app.post("/settings/password", response_class=HTMLResponse)
def settings_reset_password(
    request: Request,
    current_password: str = Form(""),
    new_password: str = Form(""),
    confirm_password: str = Form(""),
):
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    if not new_password or not confirm_password:
        return render_settings(request, password_error="Enter and confirm your new password.")
    if new_password != confirm_password:
        return render_settings(request, password_error="Passwords do not match.")

    user = get_user_by_email(current_user)
    if not user or not verify_password(current_password, user.get("password_hash", "")):
        return render_settings(request, password_error="Current password is incorrect.")

    updated = update_user_password(current_user, new_password)
    if not updated:
        return render_settings(request, password_error="Could not update password.")
    return render_settings(request, password_message="Password updated.")


@app.post("/settings/cancel-subscription", response_class=HTMLResponse)
def settings_cancel_subscription(request: Request):
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    record = get_premium_record(current_user)
    if not is_premium_active(record):
        return render_settings(request, subscription_error="No active subscription.")

    subscription_id = record.get("stripe_subscription_id") if record else None
    if not subscription_id:
        set_user_premium(current_user, False, clear_subscription=True)
        return render_settings(request, subscription_message="Subscription canceled.")

    if not STRIPE_SECRET_KEY:
        return render_settings(request, subscription_error="Subscription management is unavailable right now.")
    if STRIPE_SECRET_KEY:
        try:
            import stripe
        except Exception:
            return render_settings(request, subscription_error="Subscription management is unavailable right now.")
        stripe.api_key = STRIPE_SECRET_KEY
        try:
            subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True,
            )
        except Exception:
            return render_settings(request, subscription_error="Failed to cancel subscription.")

    period_end = None
    if subscription and subscription.get("current_period_end"):
        period_end = datetime.fromtimestamp(int(subscription["current_period_end"]), tz=timezone.utc)
    set_user_premium(
        current_user,
        True,
        subscription_id=subscription_id,
        period_end=period_end,
        cancel_at_period_end=True,
    )
    return render_settings(request, subscription_message="Cancellation scheduled.")


@app.get("/premium", response_class=HTMLResponse)
def premium_page(request: Request):
    message = ""
    error = ""
    if request.query_params.get("success") == "1":
        message = "Checkout completed. Premium will activate shortly."
    elif request.query_params.get("canceled") == "1":
        error = "Checkout canceled."
    else:
        error_code = request.query_params.get("error")
        if error_code == "stripe_config":
            error = "Checkout is unavailable right now."
        elif error_code == "stripe_import":
            error = "Checkout is unavailable right now."
        elif error_code == "stripe_session":
            error = "Could not start checkout. Please try again."

    return render_premium_page(request, message=message, error=error)


@app.post("/subscribe")
def subscribe(request: Request):
    current_user = get_current_user(request)
    if not current_user:
        return JSONResponse({"error": "not_authenticated"}, status_code=401)
    if not STRIPE_SECRET_KEY or not TIER_ONE_STRIPE_ID:
        return JSONResponse({"error": "stripe_config"}, status_code=400)
    try:
        import stripe
    except Exception:
        return JSONResponse({"error": "stripe_import"}, status_code=500)

    stripe.api_key = STRIPE_SECRET_KEY
    base_url = str(request.base_url).rstrip("/")
    return_url = f"{base_url}/premium?success=1&session_id={{CHECKOUT_SESSION_ID}}"
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            ui_mode="embedded",
            line_items=[{"price": TIER_ONE_STRIPE_ID, "quantity": 1}],
            return_url=return_url,
            customer_email=current_user,
            subscription_data={
                "metadata": {"user_email": current_user},
            },
            metadata={"user_email": current_user},
        )
    except Exception:
        return JSONResponse({"error": "stripe_session"}, status_code=500)

    client_secret = session.client_secret if session else None
    if not client_secret:
        return JSONResponse({"error": "stripe_session"}, status_code=500)
    return JSONResponse({"clientSecret": client_secret})


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        return JSONResponse({"error": "webhook_not_configured"}, status_code=400)
    try:
        import stripe
    except Exception:
        return JSONResponse({"error": "stripe_not_installed"}, status_code=500)

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, signature, STRIPE_WEBHOOK_SECRET)
    except Exception:
        return JSONResponse({"error": "invalid_signature"}, status_code=400)

    if event.get("type") == "checkout.session.completed":
        session = event.get("data", {}).get("object", {})
        email = session.get("customer_email") or session.get("metadata", {}).get("user_email")
        if email:
            subscription_id = session.get("subscription")
            period_end = None
            cancel_at_period_end = None
            if subscription_id:
                try:
                    subscription = stripe.Subscription.retrieve(subscription_id)
                    if subscription.get("current_period_end"):
                        period_end = datetime.fromtimestamp(
                            int(subscription["current_period_end"]), tz=timezone.utc
                        )
                    cancel_at_period_end = subscription.get("cancel_at_period_end")
                except Exception:
                    subscription = None
            set_user_premium(
                email,
                True,
                subscription_id=subscription_id,
                period_end=period_end,
                cancel_at_period_end=cancel_at_period_end,
            )

    if event.get("type") in {"customer.subscription.updated", "customer.subscription.deleted"}:
        subscription = event.get("data", {}).get("object", {})
        subscription_id = subscription.get("id")
        status = subscription.get("status")
        cancel_at_period_end = subscription.get("cancel_at_period_end")
        period_end = None
        if subscription.get("current_period_end"):
            period_end = datetime.fromtimestamp(int(subscription["current_period_end"]), tz=timezone.utc)

        is_active = status in ACTIVE_SUBSCRIPTION_STATUSES
        if subscription_id:
            if event.get("type") == "customer.subscription.deleted":
                update_premium_by_subscription_id(
                    subscription_id,
                    False,
                    period_end=None,
                    cancel_at_period_end=False,
                    clear_subscription=True,
                )
            else:
                update_premium_by_subscription_id(
                    subscription_id,
                    is_active,
                    period_end=period_end,
                    cancel_at_period_end=cancel_at_period_end,
                )
        else:
            email = subscription.get("metadata", {}).get("user_email")
            if email:
                set_user_premium(
                    email,
                    is_active,
                    subscription_id=subscription_id,
                    period_end=period_end,
                    cancel_at_period_end=cancel_at_period_end,
                    clear_subscription=not is_active,
                )

    return JSONResponse({"status": "ok"})


@app.post("/favorite")
async def toggle_favorite(request: Request):
    current_user = get_current_user(request)
    if not current_user:
        return JSONResponse({"error": "not_authenticated"}, status_code=401)
    if not FAVORITES_DATABASE_URL:
        return JSONResponse({"error": "favorites_unavailable"}, status_code=503)
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_payload"}, status_code=400)

    title = (payload.get("title") or "").strip()
    release_date = str(payload.get("release_date") or "")
    poster_path = str(payload.get("poster_path") or "")
    genres = str(payload.get("genres") or "")

    if not title:
        return JSONResponse({"error": "missing_title"}, status_code=400)

    try:
        exists = favorite_exists_for_user(current_user, title, release_date)
    except Exception:
        return JSONResponse({"error": "favorites_unavailable"}, status_code=503)

    if exists:
        try:
            remove_favorite_for_user(current_user, title, release_date)
        except Exception:
            return JSONResponse({"error": "favorites_unavailable"}, status_code=503)
        return JSONResponse({"favorited": False})

    is_premium = get_user_is_premium(current_user)
    if not is_premium:
        try:
            current_count = count_favorites_for_user(current_user)
        except Exception:
            return JSONResponse({"error": "favorites_unavailable"}, status_code=503)
        if current_count >= FREE_FAVORITES_LIMIT:
            return JSONResponse({"error": "favorites_limit"}, status_code=403)

    try:
        add_favorite_for_user(current_user, title, release_date, poster_path, genres)
    except Exception:
        return JSONResponse({"error": "favorites_unavailable"}, status_code=503)

    return JSONResponse({"favorited": True})


@app.get("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("session", path="/")
    return response
