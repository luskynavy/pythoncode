import sys
import plyfile

filename = 'points.ply'
if len(sys.argv) >= 2:
    args = sys.argv[1:]
    filename = args[0]
    
print filename

nbLines = 0

with open(filename, 'r') as i_file:    
    while True:
        line = i_file.readline()
        if not line:
            break
        nbLines += 1
        if nbLines < 30:
            print line + ' END'
            
with open(filename, 'r') as i_file:
    plydata = plyfile.PlyData.read(i_file)
    
print 'nb vertex : ' + str(plydata['vertex'].data.size)
    
print 'first vertex: ', plydata['vertex'].data[0]['x'], plydata['vertex'].data[0]['y'], plydata['vertex'].data[0]['z']
        
print(nbLines) 