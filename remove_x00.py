import os
for filename in os.listdir():
    if filename.endswith('txt'):
        new_filename = filename[:-4] + '_new.txt' 
        with open(filename, 'rb') as f, open(new_filename, 'wb') as g:
                context = f.read()
                context = context.replace(b'\x00', b'') 
                g.write(context)
