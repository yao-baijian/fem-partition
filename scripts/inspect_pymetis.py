import inspect
import importlib
import json
res={}
try:
    pymetis = importlib.import_module('pymetis')
    try:
        sig = inspect.signature(pymetis.part_graph)
        res['pymetis_sig'] = str(sig)
        res['pymetis_params'] = list(sig.parameters.keys())
    except Exception as e:
        res['pymetis_sig'] = f'<no signature: {e!r}>'
except Exception as e:
    res['pymetis_error'] = str(e)
print(json.dumps(res))
