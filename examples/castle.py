bx,by = 0.0, 0.0

towerX = 5
towerY = 5
towerBaseZ = 8
Clamp(lower=0.1,value = towerBaseZ)


# Bottom
Box(base=(bx,by,0),dims=(towerX,towerY,towerBaseZ))

# Mid-section
towerMidZ = 2
Clamp(lower=0.01, upper=0.5*towerBaseZ, value = towerMidZ)
towerInset = 0.5
Clamp(value=towerInset,upper=towerX/3)
Clamp(value=towerInset,upper=towerY/3)
Clamp(value = towerInset, lower=0.05*towerX)
Clamp(value = towerInset, lower=0.05*towerY,)
inX = bx+towerInset
inY = by+towerInset
ouX = bx+towerX-towerInset
ouY = by+towerY-towerInset

ww = 0.5  # window Width
wh = 1.25 # window height
wi = 0.4  # window inset
cW = 0.75
cH = 0.75
Clamp(lower=0.05*towerX, upper=(towerX-towerInset)/4, value = ww)
Clamp(lower=0.05*towerY, upper=(towerX-towerInset)/4, value = ww)
Clamp(lower=0.05*towerBaseZ,value = wh)
Clamp(lower=0.01*towerX, upper=(towerX-towerInset)/4, value = wi)
Clamp(lower=0.01*towerY, upper=(towerX-towerInset)/4, value = wi)
#Clamp(lower=0.05*towerBaseZ,value = wi)

Clamp(value=towerMidZ - wh, lower=0.05)
lH = (towerMidZ - wh) / 2 # leftover heights
lX = towerX - 2*(towerInset + wi + ww)
lY = towerY - 2*(towerInset + wi + ww)
Clamp(value=lX, lower=0.05)
Clamp(value=lY, lower=0.05)
Clamp(value=lH, lower=0.05)
midH = towerBaseZ + lH

### Midsection bottom plate
Box(base=(inX,inY,towerBaseZ),dims=(towerX-towerInset*2,towerY-towerInset*2,lH))
### Midsection outer corners
Box(base=(inX,inY,midH),dims=( wi, wi, wh))
Box(base=(ouX,ouY,midH),dims=(-wi,-wi, wh))
Box(base=(ouX,inY,midH),dims=(-wi, wi, wh))
Box(base=(inX,ouY,midH),dims=( wi,-wi, wh))
### Midsection outer slabs
Box(base=(inX+wi+ww,inY,midH),dims=( lX, wi, wh))
Box(base=(inX+wi+ww,ouY,midH),dims=( lX,-wi, wh))
Box(base=(inX,inY+wi+ww,midH),dims=( wi, lY, wh))
Box(base=(ouX,inY+wi+ww,midH),dims=(-wi, lY, wh))
### Midsection inner block
Box(base=(inX+wi+ww,inY+wi+ww,midH),dims=(lX,lY,wh))
### Midsection top plate
Box(base=(inX,inY,towerBaseZ+lH+wh),dims=(towerX-towerInset*2,towerY-towerInset*2,lH))

# Top
towerTopZ = 1.5
Clamp(value=towerTopZ, upper=0.2*towerBaseZ)
Clamp(lower=0.01,value = towerTopZ)
Box(base=(bx,by,towerBaseZ+towerMidZ),dims=(towerX,towerY,towerTopZ))

# Larger Crenellations
cZ = towerBaseZ + towerMidZ + towerTopZ
farX = bx+towerX-cW
farY = by+towerY-cW

Clamp(lower=0.1*towerX, upper=0.25*towerX,value = cW)
Clamp(lower=0.1*towerY, upper=0.25*towerY,value = cW)
Clamp(lower=0.05*towerBaseZ, upper=0.1*towerBaseZ,value = cH)

Box(base=(bx,   by,   cZ),dims=(cW, cW, cH))
Box(base=(farX, by,   cZ),dims=(cW, cW, cH))
Box(base=(bx,   farY, cZ),dims=(cW, cW, cH))
Box(base=(farX, farY, cZ),dims=(cW, cW, cH))

smallCrenellationScaleFactor = 0.4

Clamp(lower=0.2,value=smallCrenellationScaleFactor,upper=0.5)
startX = bx+cW
startY = by+cW
dx = farX - startX
dy = farY - startY
scW = cW * smallCrenellationScaleFactor
scH = cH * smallCrenellationScaleFactor
Clamp(upper=0.5*towerX/8,value = scW)

sfarX = bx+towerX-scW
sfarY = by+towerY-scW

# Smaller Crenellations
for i in range(4):
   Box(base=(startX+((i+0.5)/4.5)*(dx),by,cZ),dims=(scW,scW,scH))
for i in range(4):
   Box(base=(sfarX,startY+((i+0.5)/4.5)*dy,cZ),dims=(scW,scW,scH))
for i in range(4):
   Box(base=(startX+((i+0.5)/4.5)*(dx),sfarY,cZ),dims=(scW,scW,scH))
for i in range(4):
   Box(base=(bx,startY+((i+0.5)/4.5)*dy,cZ),dims=(scW,scW,scH))

ntowers = fix(2)
sums[0] = fix(0.0)
towerbase = 0.6*towerY
towercap = 0.8*towerY
diff = towercap-towerbase
for i in range(1, const(ntowers+1)):
    connectorL[i] = 4.0
    Clamp(value=connectorL[i], lower=2*diff)
    sums[i] = sums[i-1] + connectorL[i]
    Clamp(value=connectorL[i], lower=towerY/10)

dist = (towercap-2*0.15*towercap)/9
xshift = bx+towerX/2
towerXsmallScale = towerX/20
towerXTinyScale = towerX/30
begin = xshift-towerXsmallScale
negBegin = begin-towerXsmallScale-towerXTinyScale
posBegin = begin+towerXsmallScale+towerXTinyScale
outerwallxthick = 0.3333
innerwallxthick = 0.5
Clamp(value=outerwallxthick, lower=0.06*towerX, upper=0.25*towerX)
Clamp(value=innerwallxthick, lower=0.1*towerX, upper=0.5*towerX)

walllength = 5.6
outerwalllength = (1.2)*walllength
Clamp(value=outerwalllength, upper=towerBaseZ)
capthick = 0.8
midScaleTCap = 0.15*towercap
corners = midScaleTCap
#Clamp(value=capthick, upper=0.1*walllength)

tzk = outerwalllength+capthick
xshiftCap = xshift-towercap/2
xShiftCapNeg = xshift+towercap/2

for j in range(4):
  js[j] = 2*j*dist
  xshiftCapCorners[j] = xshiftCap+corners+dist+js[j]

midTower = diff/2
smallTowerCap = 0.1*towercap
scaledCapThick = 0.6*capthick
ybegin = by + towerY
for i in range(1, const(ntowers+1)):
    # connector
    baseYs[i] = ybegin+(i-1)*towerbase+sums[i-1]
    Box(base=(begin, baseYs[i],0.0),dims=(innerwallxthick,connectorL[i],walllength))
    Box(base=(negBegin, baseYs[i],0.0),dims=(outerwallxthick,connectorL[i],outerwalllength))
    Box(base=(posBegin, baseYs[i],0.0),dims=(outerwallxthick,connectorL[i],outerwalllength))

    # base and cap of tower
    Box(base=(xshift-towerbase/2, baseYs[i]+connectorL[i],0.0),dims=(towerbase,towerbase,outerwalllength))

    ys[i] = baseYs[i]+connectorL[i]-midTower
    Box(base=(xshiftCap, ys[i], outerwalllength),dims=(towercap,towercap,capthick))

    # 4 corners
    Box(base=(xshiftCap, ys[i], tzk),dims=(corners,corners,scaledCapThick))
    Box(base=(xShiftCapNeg-midScaleTCap, ys[i], tzk),dims=(corners,corners,scaledCapThick))
    yCaps[i] = ys[i] + towercap - corners
    Box(base=(xshiftCap, yCaps[i], tzk),dims=(corners,corners,scaledCapThick))
    Box(base=(xShiftCapNeg-corners, yCaps[i], tzk),dims=(corners,corners,scaledCapThick))

    # Smaller Crenellations

    for j in range(4):
       Box(base=(xshiftCapCorners[j], ys[i], tzk), dims=(smallTowerCap, smallTowerCap, scaledCapThick))

    ycornerDists[i] = ys[i] + corners + dist
    for j in range(4):
       Box(base=(xshiftCap, ycornerDists[i]+js[j], tzk), dims=(smallTowerCap, smallTowerCap, scaledCapThick))

    yMidScales[i] = ys[i] + midScaleTCap+dist
    for j in range(4):
       Box(base=(xShiftCapNeg-smallTowerCap, yMidScales[i]+js[j], tzk), dims=(smallTowerCap, smallTowerCap, scaledCapThick))

    yTowerCaps[i]= ys[i] + towercap - smallTowerCap
    for j in range(4):
       Box(base=(xshiftCapCorners[j],yTowerCaps[i], tzk), dims=(smallTowerCap, smallTowerCap, scaledCapThick))

# Side tower
connector = 4.0
Clamp(value=connector, lower=2*diff)
yshift = ybegin+ntowers*towerbase+sums[const(ntowers)]-towerbase/2
xShiftMidBase = xshift+towerbase/2
Box(base=(xShiftMidBase, yshift, 0.0),dims=(connector, innerwallxthick, walllength))
Box(base=(xShiftMidBase, yshift-outerwallxthick, 0.0),dims=(connector, outerwallxthick, outerwalllength))
Box(base=(xShiftMidBase, yshift+outerwallxthick, 0.0),dims=(connector, outerwallxthick, outerwalllength))

# base and cap of tower
Box(base=(xShiftMidBase+connector, yshift-towerbase/2,0.0),dims=(towerbase,towerbase,outerwalllength))
xShiftMidBaseConn = xShiftMidBase+connector-midTower
yShiftMidBase = yshift-towerbase/2-midTower
Box(base=(xShiftMidBaseConn, yShiftMidBase, outerwalllength), dims=(towercap,towercap,capthick))

# 4 corners
Box(base=(xShiftMidBaseConn, yShiftMidBase, tzk),dims=(corners,corners,scaledCapThick))
xShiftMidBaseConn2 = xShiftMidBaseConn+(towercap-corners)
Box(base=(xShiftMidBaseConn2, yShiftMidBase, tzk),dims=(corners,corners,scaledCapThick))
Box(base=(xShiftMidBaseConn, yShiftMidBase+(towercap-corners), tzk),dims=(corners,corners,scaledCapThick))
Box(base=(xShiftMidBaseConn2, yShiftMidBase+(towercap-corners), tzk),dims=(corners,corners,scaledCapThick))

# Smaller Crenellations
yShiftMidBaseScaled = yShiftMidBase +corners +dist
xShiftMidBaseConnSmall = xShiftMidBaseConn +corners +dist
for j in range(4):
    Box(base=(xShiftMidBaseConnSmall+js[j], yShiftMidBase, tzk),dims=(smallTowerCap, smallTowerCap, scaledCapThick))

for j in range(4):
    Box(base=(xShiftMidBaseConn, yShiftMidBaseScaled+js[j], tzk),dims=(smallTowerCap, smallTowerCap, scaledCapThick))

towerCapScaled = (towercap-smallTowerCap)
for j in range(4):
    Box(base=(xShiftMidBaseConnSmall+js[j], yShiftMidBase +(towerCapScaled), tzk),dims=(smallTowerCap, smallTowerCap, scaledCapThick))

for j in range(4):
    Box(base=(xShiftMidBaseConn+(towerCapScaled), yShiftMidBaseScaled+js[j], tzk),dims=(smallTowerCap, smallTowerCap, scaledCapThick))

cylinderconnector = 6.0
cylinderyshift = ybegin+ntowers*towerbase+sums[const(ntowers)]+cylinderconnector
Clamp(value=cylinderconnector, lower=2*diff)
radtower = 2.5
Clamp(value=radtower, lower=innerwallxthick+2*outerwallxthick)
C = Cylinder(subs=72, location=(xshift, cylinderyshift+radtower, (towerBaseZ+wh)/2), radius=radtower, depth=towerBaseZ+wh)
C2 = Circle(radius=1.2*radtower, subs=48, location=(xshift, cylinderyshift+radtower, towerBaseZ+wh), fill='TRIFAN')
Translate(value=(0,0,0.4*(towerBaseZ+wh)), select=C2.v([0]))
Loop(select=C2.v([1,2]))
Fill()

ang = 1.0471975511965976
cosang = cos(ang)
sinang = sin(ang)
Clamp(value=ang, lower=pi/5, upper=pi/2)
cyledge = 2*pi*radtower/72.0
wallthickness = cyledge
Clamp(value=wallthickness, lower=0.1*2*pi*radtower/72.0, upper=cyledge)
angconnector = 6.0
Clamp(value=angconnector, lower=2*(diff))

BC1 = Box(dims=(wallthickness, angconnector, walllength))
Rotate(axis='Z', center=(0,0,0), theta=ang, select=BC1.all())

towerCos = radtower +radtower*cosang
xShiftB = xshift-radtower*sinang
Translate(value=(xShiftB, cylinderyshift +towerCos, 0), select=BC1.all())
E1=Extrude(length=wallthickness, select=BC1.v([0,1,2,3]))
E2=Extrude(length=wallthickness, select=BC1.v([4,5,6,7]))
Extrude(length=(outerwalllength-walllength), select=BC1.v([1,3])+E1.v([1,3]))
Extrude(length=(outerwalllength-walllength), select=BC1.v([5,7])+E2.v([1,3]))

mangconnector = 6.0
Clamp(value=mangconnector, lower=2*diff)
BC2 = Box(dims=(wallthickness, mangconnector, walllength))
Rotate(axis='Z', center=(0,0,0), theta=-ang, select=BC2.all())
xshiftA = xshift-radtower*(-sinang)
Translate(value=(xshiftA, cylinderyshift +towerCos, 0), select=BC2.all())
E3=Extrude(length=wallthickness, select=BC2.v([0,1,2,3]))
E4=Extrude(length=wallthickness, select=BC2.v([4,5,6,7]))
Extrude(length=(outerwalllength-walllength), select=BC2.v([1,3])+E3.v([1,3]))
Extrude(length=(outerwalllength-walllength), select=BC2.v([5,7])+E4.v([1,3]))

BCcenter = Box(dims=(wallthickness, cylinderconnector, walllength))
Translate(value=(xshift, cylinderyshift-cylinderconnector, 0), select=BCcenter.all())
E5=Extrude(length=wallthickness, select=BCcenter.v([0,1,2,3]))
E6=Extrude(length=wallthickness, select=BCcenter.v([4,5,6,7]))
Extrude(length=(outerwalllength-walllength), select=BCcenter.v([1,3])+E5.v([1,3]))
Extrude(length=(outerwalllength-walllength), select=BCcenter.v([5,7])+E6.v([1,3]))

# base and cap of tower at ang
T1 = Box(dims=(towerbase,towerbase,outerwalllength))
T2 = Box(dims=(towercap,towercap,capthick))
Rotate(axis='Z', center=(0,0,0), theta=ang, select=T1.all())
Rotate(axis='Z', center=(0,0,0), theta=ang, select=T2.all())
Translate(value=(xShiftB-angconnector*sinang, cylinderyshift +towerCos +angconnector*cosang, 0), select=T1.all())
Translate(value=(-0.5*towerbase*cosang, -0.5*towerbase*sinang,0), select=T1.all())
xSinAng = xShiftB-(angconnector-diff/2)*sinang
yCosAng = (angconnector-diff/2)*cosang
smallerBaseY = cylinderyshift +towerCos +yCosAng
Translate(value=(xSinAng, smallerBaseY, 0), select=T2.all())
halfCapCosAng = -0.5*towercap*cosang
halfCapSinAng = -0.5*towercap*sinang
Translate(value=(halfCapCosAng, halfCapSinAng,0), select=T2.all())
Translate(value=(0,0,outerwalllength), select=T2.all())

dp = (0.8-0.15*0.8)*towerY
# 4 corners
for i in range(0, 2):
    for j in range(0, 2):
      Box(base=(i*dp, j*dp, 0), dims=(midScaleTCap,midScaleTCap,scaledCapThick))
      Rotate(axis='Z', center=(0,0,0), theta=ang)
      Translate(value=(xSinAng, smallerBaseY, 0))
      Translate(value=(0,0,tzk))
      Translate(value=(halfCapCosAng, halfCapSinAng,0))      
      
# Smaller Crenellations

for j in range(4):
    tbaseXY[j] = midScaleTCap+dist+js[j]
    Box(base=(0, tbaseXY[j], 0), dims=(smallTowerCap,smallTowerCap,scaledCapThick))
    Rotate(axis='Z', center=(0,0,0), theta=ang)
    Translate(value=(xSinAng, smallerBaseY, 0))
    Translate(value=(0,0,tzk))
    Translate(value=(halfCapCosAng, halfCapSinAng,0))

for j in range(4):
    Box(base=(tbaseXY[j], 0, 0), dims=(smallTowerCap,smallTowerCap,scaledCapThick))
    Rotate(axis='Z', center=(0,0,0), theta=ang)
    Translate(value=(xSinAng, smallerBaseY, 0))
    Translate(value=(0,0,tzk))
    Translate(value=(halfCapCosAng, halfCapSinAng,0))

tbaseYY = (0.8-0.1*0.8)*towerY
for j in range(4):
    Box(base=(tbaseYY, tbaseXY[j], 0), dims=(smallTowerCap,smallTowerCap,scaledCapThick))
    Rotate(axis='Z', center=(0,0,0), theta=ang)
    Translate(value=(xSinAng, smallerBaseY, 0))
    Translate(value=(0,0,tzk))
    Translate(value=(halfCapCosAng, halfCapSinAng,0))

for j in range(4):
    Box(base=(tbaseXY[j], tbaseYY, 0), dims=(smallTowerCap,smallTowerCap,scaledCapThick))
    Rotate(axis='Z', center=(0,0,0), theta=ang)
    Translate(value=(xSinAng, smallerBaseY, 0))
    Translate(value=(0,0,tzk))
    Translate(value=(halfCapCosAng, halfCapSinAng,0))


# base and cap of tower at -ang
xRotBase = xshiftA-(mangconnector-diff/2)*(-sinang)
yCosmAng = (mangconnector-diff/2)*cosang
yRotBase = cylinderyshift +towerCos +yCosmAng

T3 = Box(dims=(towerbase,towerbase,outerwalllength))
T4 = Box(dims=(towercap,towercap,capthick))
Rotate(axis='Z', center=(0,0,0), theta=-ang, select=T3.all())
Rotate(axis='Z', center=(0,0,0), theta=-ang, select=T4.all())
Translate(value=(xshiftA-mangconnector*(-sinang), cylinderyshift +towerCos +mangconnector*cosang, 0), select=T3.all())
Translate(value=(-0.5*towerbase*cosang, -0.5*towerbase*(-sinang),0), select=T3.all())
Translate(value=(xRotBase, yRotBase, 0), select=T4.all())
Translate(value=(-0.5*towercap*cosang, 0.5*towercap*sinang,0), select=T4.all())
Translate(value=(0,0,outerwalllength), select=T4.all())

halfCapNegSinang = -0.5*towercap*(-sinang)

# 4 corners
for i in range(0, 2):
    for j in range(0, 2):
      Box(base=(i*dp, j*dp, 0), dims=(midScaleTCap,midScaleTCap,scaledCapThick))
      Rotate(axis='Z', center=(0,0,0), theta=-ang)
      Translate(value=(xRotBase, yRotBase, 0))
      Translate(value=(0,0,tzk))
      Translate(value=(halfCapCosAng, halfCapNegSinang,0))

# Smaller Crenellations
for j in range(4):
    Box(base=(0, tbaseXY[j], 0), dims=(smallTowerCap,smallTowerCap,scaledCapThick))
    Rotate(axis='Z', center=(0,0,0), theta=-ang)
    Translate(value=(xRotBase, yRotBase, 0))
    Translate(value=(0,0,tzk))
    Translate(value=(halfCapCosAng, halfCapNegSinang,0))

for j in range(4):
    Box(base=(tbaseXY[j], 0, 0), dims=(smallTowerCap,smallTowerCap,scaledCapThick))
    Rotate(axis='Z', center=(0,0,0), theta=-ang)
    Translate(value=(xRotBase, yRotBase, 0))
    Translate(value=(0,0,tzk))
    Translate(value=(halfCapCosAng, halfCapNegSinang,0))

for j in range(4):
    Box(base=(tbaseYY, tbaseXY[j], 0), dims=(smallTowerCap,smallTowerCap,scaledCapThick))
    Rotate(axis='Z', center=(0,0,0), theta=-ang)
    Translate(value=(xRotBase, yRotBase, 0))
    Translate(value=(0,0,tzk))
    Translate(value=(halfCapCosAng, halfCapNegSinang,0))

for j in range(4):
    Box(base=(tbaseXY[j], tbaseYY, 0), dims=(smallTowerCap,smallTowerCap,scaledCapThick))
    Rotate(axis='Z', center=(0,0,0), theta=-ang)
    Translate(value=(xRotBase, yRotBase, 0))
    Translate(value=(0,0,tzk))
    Translate(value=(halfCapCosAng, halfCapNegSinang,0))

