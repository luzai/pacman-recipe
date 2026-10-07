"""Explicit text-only observation identities; no image placeholders."""
import hashlib
import json

def text_observation_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def text_sent_prompt_sha256(system,user):
    raw=json.dumps(dict(system=system,user=user,observation_mode='ascii'),sort_keys=True,separators=(',',':'),allow_nan=False)
    return text_observation_sha256(raw)
