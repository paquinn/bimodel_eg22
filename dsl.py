def run(program, name, locals):
    exec(compile(program, name, 'exec'), globals(), locals)