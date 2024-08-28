 #!/bin/bash 
# python benchmark_bobyqa.py -n 12 -p 1
# python benchmark_bobyqa.py -n 12 -p 2
# python benchmark_bobyqa.py -n 12 -p 3
# python benchmark_bobyqa.py -n 12 -p 4
# python benchmark_bobyqa.py -n 12 -p 5
# python benchmark_bobyqa.py -n 12 -p 1 -t budget
python benchmark_bobyqa.py -n 12 -p 2 -t budget -s 1000
python benchmark_bobyqa.py -n 12 -p 3 -t budget -s 1000
python benchmark_bobyqa.py -n 12 -p 4 -t budget -s 1000
python benchmark_bobyqa.py -n 12 -p 5 -t budget -s 1000
