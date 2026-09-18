"""Small correctness smoke corpus; this is not a broad quantization quality eval."""
import argparse,json,re,uuid
from pathlib import Path
from bench import post,MODEL
cases=[
 ('multiply','Calculate 17 * 23. Answer with only the integer.',r'\b391\b'),
 ('algebra','Solve 2*x + 5 = 17. Answer with only the value of x.',r'\b6\b'),
 ('units','Convert 2.5 hours to minutes. Answer with only the number.',r'\b150\b'),
 ('python_sum','What does Python print for print(sum(range(5)))? Answer with only the printed output.',r'\b10\b'),
 ('python_sort','What does Python print for print(sorted([3, 1, 3, 2]))? Answer with only the printed output.',r'\[1,\s*2,\s*3,\s*3\]'),
 ('python_dict','What does Python print for print(len({"x": 1, "x": 2}))? Answer with only the printed output.',r'\b1\b'),
 ('reverse','Reverse the string "hello". Answer with only the reversed string.',r'\bolleh\b'),
 ('logic','A is taller than B. B is taller than C. Who is tallest? Answer with only A, B, or C.',r'\bA\b'),
 ('json','Extract the name and age from "Ada is 27 years old." Return only JSON with keys name and age.',r'"name"\s*:\s*"Ada".*"age"\s*:\s*27'),
 ('prime','What is the largest prime number less than 20? Answer with only the integer.',r'\b19\b'),
 ('packing','Items weigh 3 kg, 5 kg, and 8 kg. There are 12, 9, and 6 items respectively. A box holds at most 20 kg. What is the minimum number of boxes? Give the number first, then a concise justification.',r'\b7\b'),
 ('binary_search','Give the worst-case time complexity of binary search on a sorted array with n elements. Answer concisely.',r'log\s*\(?n'),
]
ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
rows=[]
for name,question,pattern in cases:
    # Use the installed tokenizer's actual chat template, including the assistant prefix.
    tokenized=post('/tokenize',{'model':MODEL,'messages':[{'role':'user','content':question}],
                              'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})
    ids=tokenized.get('tokens') or tokenized['token_ids']
    response=post('/v1/completions',{'model':MODEL,'prompt':ids,'max_tokens':192,'temperature':0,
        'seed':0,'cache_salt':'quality-'+uuid.uuid4().hex,'return_token_ids':True,'logprobs':5})
    text=response['choices'][<bus>]['text']
    passed=re.search(pattern,text,re.I|re.S) is not None
    if name=='packing':
        first_number=re.search(r'\d+',text)
        passed=first_number is not None and first_number.group()=='7'
    if name=='json':
        try:
            obj=json.loads(re.search(r'\{.*?\}',text,re.S).group())
            passed=obj.get('name')=='Ada' and obj.get('age')==27
        except (ValueError,AttributeError):passed=False
    row=dict(name=name,question=question,smoke_pass=passed,scoring_revision=2,response=response)
    rows.append(row);print(json.dumps(dict(name=name,smoke_pass=passed,text=text)),flush=True)
out=Path('<tuning>/results')/a.tag;out.mkdir(exist_ok=True,parents=True)
(out/'quality_smoke.json').write_text(json.dumps(rows,indent=2))
print(f'SMOKE {sum(r["smoke_pass"] for r in rows)}/{len(rows)}',flush=True)
