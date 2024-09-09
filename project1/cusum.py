def cusum(x, mu, var, w, th, rev = False):
    '''
    cusum change detection method, ref: https://en.wikipedia.org/wiki/CUSUM
    '''

    #print(f"x: {x}")
    sz = len(x)
    assert sz>0
    assert(var!=0)
    z = (x-mu)/var
    sh = [None]*sz
    sl = [None]*sz
    count = 0 
    if rev:
        i = sz-1
        j = sz-1
        inc = -1
    else:
        i = 0
        j = 0
        inc = 1

    sh[i] = 0
    sl[j] = 0
    while(count<sz-1):
        i_ = i+inc
        sh[i_]=max(0,sh[i]+z[i_]-w)
        if (sh[i_]>th):
            break
        i = i_
        count+=1

    count = 0
    while(count<sz-1):
        j_ = j+inc
        sl[j_]=max(0,sl[j]-z[j_]-w)
        if (sl[j_]>th):
            break
        j = j_
        count+=1

    # print(sl)
    return i_, j_

def ransacBoarder(data, range):
    model_robust, inliers = ransac(
    data, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000)
    return model_robust.predict_y([np.mean(range)])