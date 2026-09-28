# Level data source

`1.txt` is an API-stable snapshot of
[`pacman-python/pacman/res/levels/1.txt` at commit
`01eff954d4ee09bcc0937b77fea7d382496e891b`](https://github.com/luzai/pacman-python/blob/01eff954d4ee09bcc0937b77fea7d382496e891b/pacman/res/levels/1.txt).

The upstream README credits David Reilly and Andy Sommerville and explicitly
permits redistribution with credit. Pacman preserves that attribution here.

Upstream raw-file SHA-256 on the source Windows checkout:
`011a82373c932e80551e0a53984d6d53a9a3ded4df200eb11fd5e7be03c5aa21`.
The public environment revision normalizes CRLF/LF before hashing and is
`36116c17c6c0805fdb1a07216357ac64c88d2c3108a0e37dce2a01b4ea2a8b97`, so the
same committed level has one identity on Windows and Linux.

The bundled file uses normalized UTF-8/LF formatting. It is retained only for
oracle route verification, not as a game backend. Its authoritative digest is
stored in `manifest.json`; production trajectories record the digest exposed by
`PygamePacmanEnv.spec.level_revision` from the external checkout.
