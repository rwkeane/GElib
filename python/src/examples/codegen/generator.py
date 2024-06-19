from typing import List, Tuple
import re
import sys
import inspect
import torch

"""
This file takes in a list containing all PyTorch operations, and generates a
base class which can be used in a SFINAE-type style to call all pytorch
operations for any give class on its instance variables.
"""

def getParameterName(param : inspect.Parameter) -> Tuple[str, bool, bool]:
  if param.kind == inspect.Parameter.VAR_POSITIONAL:
    return f"*{param.name}"  # Output: *args
  elif param.kind == inspect.Parameter.VAR_KEYWORD:
    return f"**{param.name}"  # Output: **kwargs
  else:
      return param.name  # Output: a, b
  
def getAnnotation(param : inspect.Parameter, start : str = ":") -> str:
  if param.annotation == None:
     return ""
  if not isinstance(param.annotation, str):
     return ""
  
  a = param.annotation.strip()
  if a.startswith("'") or a.startswith("\""):
    assert a.endswith("'") or a.endswith("\""), a
    a = a[1:-2]

  if a != "":          
      return " {0} '{1}'".format(start, a)
  else:
      return ""

def getVariableInfo(param : inspect.Parameter) -> str:
   ann = getParameterName(param)
   return ann[0] + getAnnotation(type, param), ann[1], ann[2]

def getReturnString(parameter : inspect.Parameter):
  annotations = parameter.__annotations__
  if not annotations.__contains__("return"):
      return ""
   
  return getAnnotation(annotations["return"], "->")

def getSignature(function):
  name = function[0]
  assert isinstance(name, str)

  sig = inspect.signature(function)
  vals = sig.parameters.values()
  
  return_annotation = getAnnotation(sig.return_annotation, "->")
  parameters = ""
  for val in vals:
    if parameters != "":
      assert call == ""
      parameters += ", "
      call += ", "
    param, args, kwargs = getVariableInfo(val)

    parameters += param

  return "  def {0}({1}){2}:\n".format(name, parameters, return_annotation)
  
def processParam(var_name : str, kwargs_name : str):
  return """
    assert not kwargs_name,__contains__(var_name)
    {1}['{0}'] = {0}
""".format(var_name, kwargs_name)

def getFunctionStart(function):
  result = getSignature(function) + "\n"
  
  sig = inspect.signature(function)
  vals = sig.parameters.values()
  
  kwargs_name = ""
  for val in vals:
    param, args, kwargs = getVariableInfo(val)
    if kwargs:
      assert kwargs_name == ""
      kwargs_name = param[2:]

  if kwargs_name == "":
    kwargs_name = "kwargs"

  for val in vals:
    param, args, kwargs = getVariableInfo(val)
    if args or kwargs:
       continue
    
    result += processParam(param, kwargs_name)

def getFunctionCall(function):
  sig = inspect.signature(function)
  vals = sig.parameters.values()
  
  kwargs_name = ""
  for val in vals:
    param, args, kwargs = getVariableInfo(val)
    if kwargs:
      assert kwargs_name == ""
      kwargs_name = param[2:]

  name = function[0]
  call = """
    results = []
    parts = self._getParts()

    # Calculate the result for each individual part.
    for i in range(len(parts)):
      part = parts[i]

      # Modify all |args| that are PointCloud instances to just be the
      # specific part we care about.
      part_args = list(args)
      for i, arg in enumerate(args):
        if isinstance(arg, self._child_type):
          part_args[i] = arg._getParts()[i]
        elif isinstance(arg, torch.Tensor):
          assert arg.size()[-2] == 1
          new_size = list(arg.size())
          new_size[-2] = part.size()[-2]
          part_args[i] = arg.expand(tuple(new_size))

      # Do the same for |{1}|.
      part_{1} = {1}.copy()
      for key, value in {1}.items():
        if isinstance(value, self._child_type):
          part_{1}[key] = value._getParts()[i]
        elif isinstance(value, torch.Tensor):
          assert value.size()[-2] == 1
          new_size = list(value.size())
          new_size[-2] = part.size()[-2]
          part_{1}[key] = arg.expand(tuple(new_size))

      # Call the underlying function with modifies parameters.
      results.append(part.{0}(*part_args, **part_{1}))

    # Turn it into an object and return it
    if results[0] == None:
      all_nones = True
      for i in range(1, len(results)):
        all_nones &= (results[i] == None)
      if all_nones:
        return None
    elif isinstance(results[0], torch.Tensor):
      return self._createObject(results)
    
    return results
  """
  return call.format(name, kwargs_name)

def writeFunction(file, function):
  string = getFunctionStart(function) + getFunctionCall(function)
  file.write(string)
      
def processMembers(file, type):
   members = inspect.getmembers(type,
                                predicate = inspect.isfunction)
   for member in members:
      writeFunction(file, member)

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

def writeFunction(file, function_name : str, kwargs_name : str = "kwargs"):
  function = """
  def {0}(self, *args, **{1}):
    # If the internals are inaccessible, treat this as a call on SO3vecArr.
    if self.can_access_internals():
      args, {1} = self.extractChildTypeInput(*args, **{1})
      return self._getVec().{0}(*args, **{1})

    # Else, try to propegate to the underlying parts as tensors.
    else:
      results = []
      parts = self._getParts()

      # Calculate the result for each individual part.
      for i in range(len(parts)):
        part = parts[i]
        
        # Clean the input
        part_args, part_{1} = self.cleanForPartsCall(part = part, *args, **{1})

        # Call the underlying function with modifies parameters.
        results.append(part.{0}(*part_args, **part_{1}))

      # Turn it into an object and return it
      if results[0] == None:
        all_nones = True
        for i in range(1, len(results)):
          all_nones &= (results[i] == None)
        if all_nones:
          return None
      elif isinstance(results[0], torch.Tensor):
        return self._createObject(results)
      
      return results
  """
  file.write(function.format(function_name, kwargs_name))

def writeHeader(f):
  str = """from typing import List
import torch

from .tensor_recurser_client import TensorRecurserClient

class TensorRecurser:
  \"""
   To use this class, the child must define the following functions:
   1. self._getVec() to return the SO3vecArr / SO3vec for this instance.
   2. self._createObject(results) to create a new instance of the object to be
      returned to the caller.
   3. can_access_internals() to check if the internals of this object can be
      accessed.

   NOTE: This file is autogenerated programmatically. Do not make manual changes
   as they won't persist.
  \"""

  def __init__(self, child_type = TensorRecurserClient, *args, **kwargs):
    \"""
    |child_type| is the type of the child that extends this parent, to allow
    this class to act in a SFINAE manner.
    \"""
    self._child_type = child_type

  def _getParts(self) -> List[torch.Tensor]:
    return self._getVec().parts

  def extractChildTypeInput(self, *args, **kwargs):
    # Modify all |args| that are PointCloud instances to just be the
    # specific part we care about.
    args = list(args)
    for i, arg in enumerate(args):
      if isinstance(arg, self._child_type):
        args[i] = arg._getVec()

    # Do the same for |kwargs|.
    kwargs = kwargs.copy()
    for key, value in kwargs.items():
      if isinstance(value, self._child_type):
        kwargs[key] = value._getVec()

    return args, kwargs

  def cleanForPartsCall(self, part, *args, **kwargs):
    # Modify all |args| that are PointCloud instances to just be the
    # specific part we care about.
    args = list(args)
    for i, arg in enumerate(args):
      if isinstance(arg, self._child_type):
        args[i] = arg._getParts()[i]
      elif isinstance(arg, torch.Tensor):
        assert arg.size()[-2] == 1
        new_size = list(arg.size())
        new_size[-2] = part.size()[-2]
        args[i] = arg.expand(tuple(new_size))

    # Do the same for |kwargs|.
    kwargs = kwargs.copy()
    for key, value in kwargs.items():
      if isinstance(value, self._child_type):
        kwargs[key] = value._getParts()[i]
      elif isinstance(value, torch.Tensor):
        assert value.size()[-2] == 1
        new_size = list(value.size())
        new_size[-2] = part.size()[-2]
        kwargs[key] = arg.expand(tuple(new_size))

    return args, kwargs

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

# if __name__=="__main__": 
#     command_args = sys.argv
#     assert len(command_args) == 3, len(command_args)

#     source = command_args[1]
#     dest = command_args[2]

#     print("Source:", source, " Dest:", dest)

#     source = open(source, "r")
#     dest = open(dest, "x")  # TODO: w not x

#     writeHeader(dest)
#     processMembers(dest, torch.Tensor)