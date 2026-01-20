Auth hashing notes
------------------

Change: the project hashing backend was switched to `argon2` via Passlib's
`CryptContext` in these modules:

- `app/auth/hashing.py`
- `app/auth/utils.py`
- `app/auth/security.py`

Why: bcrypt has a 72-byte input limit which raises a ValueError for long
passwords. `argon2` has no such limitation and is generally recommended for
password hashing for modern applications.

Install:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Notes / alternatives:
- If you prefer bcrypt semantics while avoiding the 72-byte truncation issue,
  use `bcrypt_sha256` in `CryptContext` instead of `argon2` (Passlib will
  pre-hash using SHA-256 before bcrypt).
- Keep `argon2-cffi` in `requirements.txt` to ensure the backend is available
  in development and deployment environments.
