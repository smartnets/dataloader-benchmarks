#!/bin/sh
gpg --quiet --batch --yes --decrypt --passphrase="$WHEEL_PASSWORD" \
--output infrastructure/indra-0.0.5-py3-none-any.whl infrastructure/indra-0.0.5-py3-none-any.whl.gpg
