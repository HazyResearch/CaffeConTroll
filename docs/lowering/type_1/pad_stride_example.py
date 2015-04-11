import sys

########################################################
# Small Examples
########################################################

D = []
D.append(['a', 'b', 'c', 'd'])
D.append(['e', 'f', 'g', 'h'])
D.append(['i', 'j', 'k', 'l'])
D.append(['m', 'n', 'o', 'p'])

n = 4
k = 2
d = 1
b = 1
o = 1

# First no padding, stride=1
p = 0
s = 1
m = (n + 2*p - k)/s + 1
print 'p = ' + str(p)
print 's = ' + str(s)
print 'm = ' + str(m)
print ''
D_lowered = []
for Dl_r in range(m*m*b):
    D_lowered.append([])
for bi in range(b):
    for r in range(m):
        for c in range(m):
            current_row = []
            for Dr in range(k):
                for Dc in range(k):
                    current_row.append(D[r+Dr][c+Dc])
            D_lowered[bi*m*m + r*m + c] = current_row
for Dl_r in range(len(D_lowered)):
    for Dl_c in range(len(D_lowered[Dl_r])):
        sys.stdout.write(D_lowered[Dl_r][Dl_c])
        sys.stdout.write(" & ")
    sys.stdout.write("\\\\\n")
print ''
print ''

sys.exit(0)

# stride=2
p = 0
s = 2
m = (n + 2*p - k)/s + 1
print 'p = ' + str(p)
print 's = ' + str(s)
print 'm = ' + str(m)
print ''
D_lowered = []
for Dl_r in range(m*m*b):
    D_lowered.append([])
for bi in range(b):
    for r in range(m):
        for c in range(m):
            current_row = []
            for Dr in range(k):
                for Dc in range(k):
                    current_row.append(D[r*s+Dr][c*s+Dc])
            D_lowered[bi*m*m + r*m + c] = current_row
for Dl_r in range(len(D_lowered)):
    for Dl_c in range(len(D_lowered[Dl_r])):
        sys.stdout.write(D_lowered[Dl_r][Dl_c])
        sys.stdout.write(" & ")
    sys.stdout.write("\\\\\n")
print ''
print ''

# stride=2, padding 1
p = 1
s = 2
m = (n + 2*p - k)/s + 1
print 'p = ' + str(p)
print 's = ' + str(s)
print 'm = ' + str(m)
print ''
D_lowered = []
for Dl_r in range(m*m*b):
    D_lowered.append([])
for bi in range(b):
    for r in range(m):
        for c in range(m):
            current_row = []
            for Dr in range(k):
                if r*s-p+Dr >= 0 and r*s-p+Dr < len(D):
                    for Dc in range(k):
                        if c*s-p+Dc >= 0 and c*s-p+Dc < len(D[r*s-p+Dr]):
                            current_row.append(D[r*s-p+Dr][c*s-p+Dc])
                        else:
                            current_row.append('0')
                else:
                    for Dc in range(k):
                        current_row.append('0')
            D_lowered[bi*m*m + r*m + c] = current_row
for Dl_r in range(len(D_lowered)):
    for Dl_c in range(len(D_lowered[Dl_r])):
        sys.stdout.write(D_lowered[Dl_r][Dl_c])
        sys.stdout.write(" & ")
    sys.stdout.write("\\\\\n")
print ''
print ''

# stride=2, padding 1
p = 1
s = 3
m = (n + 2*p - k)/s + 1
print 'p = ' + str(p)
print 's = ' + str(s)
print 'm = ' + str(m)
print ''
D_lowered = []
for Dl_r in range(m*m*b):
    D_lowered.append([])
for bi in range(b):
    for r in range(m):
        for c in range(m):
            current_row = []
            for Dr in range(k):
                if r*s-p+Dr >= 0 and r*s-p+Dr < len(D):
                    for Dc in range(k):
                        if c*s-p+Dc >= 0 and c*s-p+Dc < len(D[r*s-p+Dr]):
                            current_row.append(D[r*s-p+Dr][c*s-p+Dc])
                        else:
                            current_row.append('0')
                else:
                    for Dc in range(k):
                        current_row.append('0')
            D_lowered[bi*m*m + r*m + c] = current_row
for Dl_r in range(len(D_lowered)):
    for Dl_c in range(len(D_lowered[Dl_r])):
        sys.stdout.write(D_lowered[Dl_r][Dl_c])
        sys.stdout.write(" & ")
    sys.stdout.write("\\\\\n")
print ''
print ''


########################################################
# Big example
########################################################

D = []
D.append(['a', 'b', 'c', 'd', 'e'])
D.append(['f', 'g', 'h', 'i', 'j'])
D.append(['k', 'l', 'm', 'n', 'o'])
D.append(['p', 'q', 'r', 's', 't'])
D.append(['u', 'v', 'w', 'x', 'y'])

n = 5
k = 3
d = 1
b = 1
o = 1

# stride=2, padding 2, 5x5
p = 2
s = 2
m = (n + 2*p - k)/s + 1
print 'p = ' + str(p)
print 's = ' + str(s)
print 'm = ' + str(m)
print ''
D_lowered = []
for Dl_r in range(m*m*b):
    D_lowered.append([])
for bi in range(b):
    for r in range(m):
        for c in range(m):
            current_row = []
            for Dr in range(k):
                if r*s-p+Dr >= 0 and r*s-p+Dr < len(D):
                    for Dc in range(k):
                        if c*s-p+Dc >= 0 and c*s-p+Dc < len(D[r*s-p+Dr]):
                            current_row.append(D[r*s-p+Dr][c*s-p+Dc])
                        else:
                            current_row.append('0')
                else:
                    for Dc in range(k):
                        current_row.append('0')
            D_lowered[bi*m*m + r*m + c] = current_row
for Dl_r in range(len(D_lowered)):
    for Dl_c in range(len(D_lowered[Dl_r])):
        sys.stdout.write(D_lowered[Dl_r][Dl_c])
        sys.stdout.write(" & ")
    sys.stdout.write("\\\\\n")
print ''
print ''

