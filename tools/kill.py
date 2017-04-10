#!/usr/bin/env python
"""this tool is used to kill mxnet related program
in the localhost and remote hosts.
This tool now is in latest mxnet
Example: python kill-mxnet.py hosts ec2-user python
"""
from __future__ import print_function
import os
import sys
import subprocess

if len(sys.argv) != 4:
    print("usage: %s <hostfile> <user> <prog>" % sys.argv[0])
    sys.exit(1)

# host file
host_file = sys.argv[1]
# user for the process
user = sys.argv[2]
# the program you want to kill
prog_name = sys.argv[3]

kill_cmd = (
    "ps aux | "
    "grep -v grep | "
    "grep '" + prog_name + "' | "
                           "awk '{if($1==\"" + user + "\")print $2;}' | "
                                                      "xargs kill -9"
)
print(kill_cmd)

# Kill program on remote machines
with open(host_file, "r") as f:
    for host in f:
        if ':' in host:
            host = host[:host.index(':')]
        print(host)
        subprocess.Popen(["ssh", "-oStrictHostKeyChecking=no", "%s" % host, kill_cmd],
                         shell=False,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
        print("Done killing")

# Kill program on local machine
os.system(kill_cmd)
