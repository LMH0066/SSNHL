def choosed(n, d):
    if n <=100:
        if d/n <=0.02:
            y = 1
        else:
            y = 0
    elif n<=200:
        if d/n <=0.06:
            y = 1
        else:
            y = 0
    elif n<=300:
        if d/n <=0.26:
            y = 1
        else:
            y = 0
    elif n <=400:
        if d/n <=0.28:
            y = 1
        else:
            y = 0
    elif n<=500:
        if d/n <=0.34:
            y = 1
        else:
            y = 0
    else:
        if d/n <=0.38:
            y = 1
        else:
            y = 0
    return y