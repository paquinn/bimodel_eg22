rad= 1.0 
cx, cy, cz = 0.0, 0.0, 0.0
scale=2.0
S1 = Sphere(radius=rad, location=(cx, cy, cz), segments=20, ring_count=16)
# delete bottom half and segments that run from pi/2 through pi/2+pi
Resize(value=(scale, 1.0, 1.0))
S2 = Sphere(radius=0.8*rad, location=(cx, cy, cz), segments=20, ring_count=16)
Resize(value=(scale, 1.0, 1.0))
# delete bottom half and segments that run from pi/2 through pi/2+pi
# connect two spheres
sole_thick = 0.2
# FrobtSole = Extrude(length=sole_thick, )
shoe_length= 3.0
# Body = Extrude(length=shoe_length, )
# LoopCut(number_cuts=11, select=Edge from Body)
heal_height = 1.0
Cylinder(radius=sphere_r, location=(cx+shoe_length, cy, cz+heal_height), depth=sole_thick, vertices=20, end_fill_type='NGON')
# Delete left half of cylinder
for i in range(1, 11): 
	Translate(value=(0.0, 0.0, heal_height+sole_thick/2-(heal_height+sole_thick/2)*i**2/11**2), select = loopi)
ExtrudeFace(length=(heal_height+sole_thick/2), select=bottom face of half cylinder)