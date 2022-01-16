soleWidth = 5.0
soleHeight = 4.0
soleLength = 10.0


Heel = Cylinder(radius=soleWidth/2, depth=soleHeight, location=(0, 0, soleHeight/2), subs=20)
Select(all="DESELECT")

for i in range(22, 39, 2):
    Select(select=Heel.v([i, i+1]), union=True)
Delete()

for i in range(2):
    for j in range(0, 21, 2):
        Select(select=Heel.v([i+j]), union=True)
    Fill()
    Select(all="DESELECT")

Fill(select=Heel.v([0, 1, 20, 21]))
Loop(select=Heel.v([1, 3]))
soleThickness = 0.6
SoleTop = Extrude(length=soleThickness)
heelTaper = 0.75
Loop(select=Heel.v([0, 2]))
Resize(value=(heelTaper, heelTaper, 1), center=(0, 0, 0))

toeThickness = 0.4
toeLength = 1.8
ToeInner = Sphere(radius=soleWidth/2, segments=20, ring_count=16, location=(-soleLength, 0, soleThickness))
Resize(value=(toeLength, 1, 1), center=(-soleLength, 0, soleThickness))
Flip()

ToeOutter = Sphere(radius=soleWidth/2+toeThickness, segments=20, ring_count=16, location=(-soleLength, 0, soleThickness))
Resize(value=(toeLength, 1, 1), center=(-soleLength, 0, soleThickness))

Select(all='DESELECT')
for i in range(7):
    Loop(select=ToeInner.v([i+8, i+293]))
    Loop(select=ToeOutter.v([i+8, i+293]), union=True)
    Delete()
Delete(select=ToeInner.v([301])+ToeOutter.v([301]))

for i in range(9):
    Path(select=ToeInner.v([(i+11)*15, (i+11)*15+7]))
    Delete()
    Path(select=ToeOutter.v([(i+11)*15, (i+11)*15+7]))
    Delete()

Loop(select=ToeOutter.v([6, 7]))
Loop(select=ToeInner.v([6, 7]), union=True)
Fill()
Loop(select=ToeOutter.v([22, 7]))
Fill()

ToeBottom = Extrude(length=soleThickness)

Select(all='DESELECT')
Path(select=ToeBottom.v([9, 20]))
Select(select=ToeBottom.v([12, 18]), union=True)
for i in range(11):
    Select(select=ToeInner.v([15*i + 7]), union=True)
DeleteFace()

Path(select=ToeBottom.v([9, 20]))
Select(select=ToeBottom.v([12, 18]), union=True)
Fill()
Select(all='DESELECT')
for i in range(11):
    Select(select=ToeInner.v([15*i + 7]), union=True)
Fill()

divs = fix(10)
## Commented out lines denote a program change. Existing function
## only works with divs = 10, so we use a logistic curve to get
## as many divisions as we want.
#divs = fix(20)
soleDiv = soleLength / divs
Select(select=Heel.v([1, 21])+SoleTop.v([0, 10]))
offset = 20
#offset=102.1
e = fix(2.7128)
k = 15.0
mid = divs/2 - 1
for i in range(const(divs)-1):
    Sole[i] = Extrude(length=soleDiv)
    zOffset[i] = (i+1)/(10)
    zOffsetSmooth[i] = 3*pow(zOffset[i], 2) - 2*pow(zOffset[i], 3)
    #zOffset[i] = (i+1.0)/(divs) - 0.5
    #zOffsetSmooth[i] = 1 / (1 + e**(-k*zOffset[i]))
    scale[i] = (pow(i-mid, 2)+offset) / (mid**2 + offset)



for i in range(const(divs)-1):
    Resize(value=(1.0, scale[i], 1.0), select=Sole[i].all())
    Translate(value=(0, 0, -soleHeight*zOffsetSmooth[i]), select=Sole[i].all())

Bridge(select=ToeInner.v([157, 7])+ToeBottom.v([12, 18]), union=True)
