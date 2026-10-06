"""Exercise installed-wheel CLI against local protocol fixtures, never real models."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Thread


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--python',default=sys.executable)
    args=parser.parse_args()
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path,body))
            if self.path=='/api/embed':
                assert body['truncate'] is False
                result={'embeddings':[[1,len(t)] for t in body['input']]}
            elif self.path=='/v1/systemone':
                result={'answers':{k:{'type':'noul','noul':.95} for k in body['questions']}}
            else:
                self.send_error(404);return
            data=json.dumps(result).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            def run(*arguments):
                return subprocess.check_output([args.python,'-I',*arguments],cwd=path,text=True,encoding='utf-8')
            # -I and a foreign working directory prevent checkout imports.
            location=run('-c','import zero_index; print(zero_index.__file__)').strip()
            assert 'site-packages' in location.lower(),location
            run('-m','zero_index','init','-o','local.json','--effort','low')
            settings=json.loads((path/'local.json').read_text())
            for name in ('embedding','decision'):
                settings[name]['base_url']=f'http://127.0.0.1:{server.server_port}'
            (path/'local.json').write_text(json.dumps(settings),encoding='utf-8')
            source='# Example\r\n\r\nFirst source paragraph.\r\n\r\n## Details\r\n\r\nSecond source paragraph.'
            (path/'source.md').write_bytes(source.encode())
            run('-m','zero_index','index','source.md','--config','local.json','-o','tree.json')
            assert all(endpoint=='/api/embed' for endpoint,_ in calls),'EE indexing called the decision model'
            run('-m','zero_index','search','tree.json','Find evidence','--config','local.json','--hybrid','-o','result.json')
            result=json.loads((path/'result.json').read_text(encoding='utf-8'))
            assert result['variant']=='EEJ' and result['mode']=='hybrid' and result['paragraphs']
            for row in result['paragraphs']:assert row['text']==source[row['start']:row['end']]
            assert any(endpoint=='/v1/systemone' for endpoint,_ in calls)
            assert 'scripts' not in run('-c','import zero_index; from zero_index.providers import DecisionModel; import sys; print([m for m in sys.modules if m == "scripts" or m.startswith("scripts.")])')
            print(json.dumps({'installed_package':location,'passed':True,'protocol_fixture_requests':len(calls),'real_model_calls':0}))
    finally:
        server.shutdown();server.server_close();thread.join()


if __name__=='__main__':main()
