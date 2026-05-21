import inspect
import kahip
try:
    s = inspect.signature(kahip.kaffpa)
except Exception as e:
    s = f"<no signature: {e!r}>"

d = kahip.kaffpa.__doc__ or ''
attrs = [a for a in dir(kahip) if 'kaffpa' in a.lower() or 'part' in a.lower()]
with open('c:/project/fem-partition/kahip_sig.txt', 'w', encoding='utf8') as f:
    f.write('SIGNATURE: ' + str(s) + "\n\nREPR:\n" + repr(kahip.kaffpa) + "\n\nDOCSTRING:\n" + d + "\n\nATTRS:\n" + str(attrs))
print('wrote kahip_sig.txt')
