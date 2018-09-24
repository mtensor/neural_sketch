def g():
    for i in range(10):
        hit = i == 4
        yield i
        if hit: break

