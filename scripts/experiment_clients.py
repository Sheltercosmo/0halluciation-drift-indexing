"""Audited structured Codex calls; compatible with the frozen reader prompt/cache."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from urllib.request import getproxies

from scripts.bounded_clients import (Codex, path_lock, save, signature,
                                     validate_answers, validate_rankings)


class StructuredCodex(Codex):
    """Preserve raw responses and usage even when output validation fails."""
    def structured(self, instruction, payload, schema, validator, stage):
        prompt=instruction+'\nINPUT:\n'+json.dumps(payload,ensure_ascii=False)
        key=signature({'model':self.model,'reasoning_effort':'low','prompt':prompt,'schema':schema})
        cached=self.cache/(key+'.json')
        with path_lock(cached):
            return self._invoke([payload],stage,schema,prompt,key,cached,validator)

    def _execute(self,cases,mode,schema,prompt,key,cached):
        validator=(lambda value:validate_answers(value,[x['id'] for x in cases])) if mode=='reader' else (
            lambda value:validate_rankings(value,{x['id']:len(x['candidates']) for x in cases}))
        return self._invoke(cases,mode,schema,prompt,key,cached,validator)

    def _invoke(self,cases,mode,schema,prompt,key,cached,validator):
        if cached.exists():
            value=json.loads(cached.read_text(encoding='utf-8'))
            validator(value)
            return value
        prior=0
        if self.audit.path.exists():
            prior=sum(r.get('request_sha256')==key and r.get('error_type') in ('ValueError','JSONDecodeError')
                      for r in (json.loads(line) for line in self.audit.path.read_text(encoding='utf-8').splitlines()))
        if prior>=2:
            raise ValueError('Registered structural retry limit exhausted for this request')
        for attempt in range(prior,2):
            self.budget.reserve('codex_calls')
            attempt_id=uuid.uuid4().hex
            prefix=self.work/(key+'-'+attempt_id)
            schema_path=Path(str(prefix)+'-schema.json')
            answer_path=Path(str(prefix)+'-answer.json')
            save(schema_path,schema)
            env=os.environ.copy()
            for variable in ('GEMINI_API_KEY','TYPESAFE_API_KEY','OPENROUTER_API_KEY'):
                env.pop(variable,None)
            for scheme,proxy in getproxies().items():
                if scheme in ('http','https'):
                    env[scheme.upper()+'_PROXY']=proxy
            args=[self.binary,'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
                  '--sandbox','read-only','--cd',str(self.work.resolve()),'--model',self.model,
                  '-c','model_reasoning_effort="low"','--output-schema',str(schema_path.resolve()),
                  '--output-last-message',str(answer_path.resolve()),'--json','-']
            row={'stage':mode,'model':self.model,'reasoning_effort':'low','request_sha256':key,
                 'attempt_id':attempt_id,'attempt':attempt+1,'case_ids':[x.get('id') for x in cases],
                 'status':'failed'}
            started=time.perf_counter()
            try:
                process=subprocess.run(args,input=prompt,text=True,encoding='utf-8',errors='replace',
                                       capture_output=True,env=env,timeout=180)
                events=[json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')]
                complete=[e for e in events if e.get('type')=='turn.completed']
                if complete:
                    row['usage']=complete[-1].get('usage',{})
                tool_items=[e for e in events if e.get('item',{}).get('type') in {
                    'command_execution','file_change','file_changes','mcp_tool_call','web_search','tool_call'}]
                row['tool_items']=len(tool_items)
                if process.returncode or tool_items or not complete or not answer_path.exists():
                    raise RuntimeError('Codex call failed or used tools; inspect sanitized audit')
                raw=answer_path.read_text(encoding='utf-8')
                archive=self.output/'codex-responses'/(attempt_id+'.json')
                archive.parent.mkdir(parents=True,exist_ok=True)
                archive.write_text(raw,encoding='utf-8',newline='\n')
                row['response_file']=archive.relative_to(self.output).as_posix()
                value=json.loads(raw)
                validator(value)
                row['status']='ok'
                save(cached,value)
                return value
            except (ValueError,json.JSONDecodeError) as exc:
                row['error_type']=type(exc).__name__
                if attempt:
                    raise
            except BaseException as exc:
                row['error_type']=type(exc).__name__
                raise
            finally:
                row['seconds']=time.perf_counter()-started
                self.audit.append(row)
        raise RuntimeError('Structured response validation exhausted')
