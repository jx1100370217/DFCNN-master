import sys
import os
while True:
	gpu_con = os.popen("nvidia-smi | tail -n 2 | head -n 1")
	flist = gpu_con.read()
	fstr = "".join(flist)
	pid = fstr.split()[2]
	#print(pid)
	cmd = 'kill -9 ' + pid
	if pid != 'running':
		os.system(cmd)
	else:
		break
	
