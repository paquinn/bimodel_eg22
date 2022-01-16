rad1 = 2.0
dep = 0.4
cx, cy, cz = 0.0, 0.0, 0.0
Cylinder(subs=30, radius=(1.5 * rad1), depth=dep, location=(cx, cy, (cz + (2 * dep))), base_fill='NGON')
Cylinder(subs=30, radius=(1.2 * rad1), depth=dep, location=(cx, cy, (cz + dep)), base_fill='NGON')
Cylinder(subs=30, radius=rad1, depth=dep, location=(cx, cy, cz), base_fill='NGON')
nrods= 5
rodlen = 1.0
wid= 0.98
for i in range(1, (5+ 1)):
	Cylinder(subs=5, radius=(0.01 * rad1), depth=i*rodlen, location=((cx+0.7*rad1*cos((i-1)*2*pi/5), cy+0.7*rad1*sin((i-1)*2*pi/5), cz-dep/2-i/2-(0.05*rad1*i-0.05*rad1*i*cos(3*pi/16)))), base_fill='NGON')
	Sphere(radius=0.05*rad1*i, segments=16, ring_count=16, location=((cx+0.7*rad1*cos((i-1)*2*pi/5), cy+0.7*rad1*sin((i-1)*2*pi/5), cz -dep/2-i-0.05*rad1*i))) 
	Translate(value=(0,0,-(0.05*rad1*i-0.05*rad1*i*cos(3*pi/16))))
	# Delete bottom 3 rings
	Sphere(radius=0.05*rad1*i, segments=16, ring_count=16, location=((cx+0.7*rad1*cos((i-1)*2*pi/5), cy+0.7*rad1*sin((i-1)*2*pi/5), cz -dep/2-i-3*0.05*rad1*i))) 
	Translate(value=(0,0,0.05*rad1*i-0.05*rad1*i*cos(3*pi/16)))
	# Delete top 3 rings
	# for loop to scale the remaining rings by some other factor p

