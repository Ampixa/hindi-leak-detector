"""Read-only loopback review of the frozen fresh-sentence comparison."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from hindi_detector.wordlab_server import audio_byte_range

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8782)
    options = parser.parse_args()
    plan = ROOT / 'data/cross_system_sentences_v80'
    results = ROOT / 'reports/cross_system_results_v80'
    expected = {j['id']: j for j in json.loads((plan / 'selection.json').read_text())['jobs']}
    assets = {}
    for system, directory in [('pocket_teacher_250000_24l', plan / 'teacher'), ('indicf5_stock', plan / 'indicf5')]:
        for line in (directory / 'items.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['id'] not in expected or row['system'] != system or row['text'] != expected[row['id']]['text']:
                raise ValueError('Review asset is outside the frozen sentence plan')
            if row['status'] == 'ok_unreviewed':
                path = (directory / row['audio']).resolve(strict=True)
                if not path.is_relative_to((directory / 'audio').resolve()):
                    raise ValueError('Generated asset path escaped its owned audio directory')
                assets[system, row['id']] = path, row['sha256'], row['bytes']

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            LOG.info('%s ' + fmt, self.client_address[0], *values)

        def send(self, code, body, content_type, *, ranged=False):
            headers = {}
            if ranged:
                code, body, headers = audio_byte_range(body, self.headers.get('Range'))
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for k, value in headers.items():
                self.send_header(k, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            try:
                if route == '/':
                    self.send(200, (ROOT / 'review/cross-sentences.html').read_bytes(), 'text/html; charset=utf-8')
                elif route == '/api/index':
                    payload = {'sentences': json.loads((results / 'index.json').read_text()),
                        'report': json.loads((results / 'report.json').read_text()) if (results / 'report.json').exists() else None,
                        'status': json.loads((results / 'status.json').read_text())}
                    self.send(200, json.dumps(payload, ensure_ascii=False).encode(), 'application/json; charset=utf-8')
                elif route.startswith('/api/case/'):
                    identifier = route.removeprefix('/api/case/')
                    index = json.loads((results / 'index.json').read_text())
                    matches = [r for r in index if r['id'] == identifier]
                    if len(matches) != 1:
                        self.send(404, b'Unknown sentence', 'text/plain'); return
                    body = (results / 'cases' / (identifier + '.json')).read_bytes()
                    if hashlib.sha256(body).hexdigest() != matches[0]['case_sha256']:
                        raise ValueError('Sentence result differs from its recorded hash')
                    self.send(200, body, 'application/json; charset=utf-8')
                elif route.startswith('/audio/'):
                    parts = route.removeprefix('/audio/').split('/')
                    if len(parts) != 2 or tuple(parts) not in assets:
                        self.send(404, b'Unknown generated audio', 'text/plain'); return
                    path, digest, size = assets[tuple(parts)]
                    body = path.read_bytes()
                    if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
                        raise ValueError('Generated audio differs from its frozen receipt')
                    self.send(200, body, 'audio/wav', ranged=True)
                else:
                    self.send(404, b'Unknown review route', 'text/plain')
            except (BrokenPipeError, ConnectionResetError):
                LOG.info('Client closed review/audio connection: %s', route)
            except Exception:
                LOG.exception('Fresh-sentence review failed explicitly for %s', route)
                self.send(500, b'Review data validation failed; inspect owned service log.', 'text/plain')

    server = ThreadingHTTPServer(('127.0.0.1', options.port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
