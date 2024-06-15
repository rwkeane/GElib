from typing import List
import re
import sys

"""
This file takes in a list containing all PyTorch operations, and generates a
base class which can be used in a SFINAE-type style to call all pytorch
operations for any give class on its instance variables.
"""

def getData(file : str, prefix = "Tensor") -> List[str]:
  Lines = file.readlines()
  
  # Strips the newline character
  non_empty_lines = []
  for line in Lines:
      line = line.strip()
      if line != "":
         non_empty_lines.append(line)

  functions = []
  regex = "^{0}[.](.*)$".format(prefix)
  for line in non_empty_lines:
    regex_result = re.search(regex, line)
    if regex_result == None:
       continue
    
    function = regex_result.group(1)
    functions.append(function)

  return functions

def writeFunction(f, function_name : str):
  f.write("  def {0}(self, *args, **kwargs) -> List:\n".format(function_name))
  f.write("    results = [ t.{0}(*args, **kwargs) for t in self.getParts() ]\n"\
          .format(function_name))
  f.write("    return self.createObject(results)\n\n")

def writeHeader(f):
  str = """from typing import List

class TensorRecurser:
  \"""
   To use this class, the child must define the following functions:
   1. self.getParts() to return the tensor-like objects to iterate over.
   2. self.createObject(results) to create a new instance of the object to be
      returned to the caller.
  \"""

  def __init__(self, *args, **kwargs):
    pass

"""

  f.write(str)

if __name__=="__main__": 
    command_args = sys.argv
    assert len(command_args) == 3, len(command_args)

    source = command_args[1]
    dest = command_args[2]

    print("Source:", source, " Dest:", dest)

    source = open(source, "r")
    dest = open(dest, "x")  # TODO: w not x

    functions = getData(source)

    writeHeader(dest)
    for func in functions:
       writeFunction(dest, func)
