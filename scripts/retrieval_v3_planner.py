"""Isolated shared evidence planning, using the authorized Codex account allowance."""
from scripts.experiment_clients import StructuredCodex
from scripts.planned_evidence_trial import PLAN_INSTRUCTION,PLAN_SCHEMA,validate_plan


class SharedPlanner(StructuredCodex):
    def plan_batch(self,cases,docs):
        inputs=[{'id':c['id'],'question':c['question'],'title':docs[c['doc_id']]['title'],
                 'headings':list(dict.fromkeys(u['heading'] for u in docs[c['doc_id']]['units']))} for c in cases]
        item={'type':'object','properties':{'id':{'type':'string'},'needs':PLAN_SCHEMA['properties']['needs']},
              'required':['id','needs'],'additionalProperties':False}
        schema={'type':'object','properties':{'plans':{'type':'array','items':item}},
                'required':['plans'],'additionalProperties':False}
        def validate(value):
            rows=value['plans']
            if len(rows)!=len(cases) or {r['id'] for r in rows}!={c['id'] for c in cases}:
                raise ValueError('Planner changed question identities')
            for row in rows:validate_plan(row)
        value=self.structured(PLAN_INSTRUCTION+' Each case is independent. Return one plan for every exact input id.',
                              {'cases':inputs},schema,validate,'planner_batch')
        return {r['id']:r['needs'] for r in value['plans']}
