# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import (Any, List)

import oneflow as flow

from ..operation import OneflowOperation

#todo figure out what is it used for
mem_format = [
    'flow.contiguous_format',      # 0
    'flow.preserve_format',        # 1
    'flow.channels_last',          # 2
]

# this snippet is copied from torch/onnx/symbolic_helper.py,
# the original definition is in c10/core/ScalarType.h
# This indicates each scalar type's corresponding
scalar_type_to_pytorch_type = [
    'flow.uint8',        # 0
    'flow.int8',         # 1
    #'flow.short',        # 2
    'flow.int',          # 3
    'flow.int64',        # 4
    'flow.half',         # 5
    'flow.float',        # 6
    'flow.double',       # 7
#    'flow.complex32',    # 8
#    'flow.complex64',    # 9
#    'flow.complex128',   # 10
#    'flow.bool',         # 11
]

class NoOpIdentity(OneflowOperation):
    """
    this operator type is added by us
    """
    _ori_type_name = ['noop_identity']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {", ".join(inputs)}'

class ModuleOperator(OneflowOperation):
    _ori_type_name = ['ModuleOperator', 'shared']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = self.{field}({", ".join(inputs)})'

class FunctionalOperator(OneflowOperation):
    _ori_type_name = ['FunctionalOperator']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        func_name = self.type[len('Function.'):]

        if not hasattr(flow, func_name) and not hasattr(flow.nn, func_name[len('nn.'):]):
            raise RuntimeError('For now, we only support calling independent functions from `flow`, '
                               f'{func_name} is not in it.')
        return f'{output} = flow.{func_name}({", ".join(inputs)})'


class PrimConstant(OneflowOperation):
    _ori_type_name = ['prim::Constant']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # TODO: refactor this part, maybe we can remove the code gen of prim::Constant
        # TODO: deal with all the types
        if self.parameters['type'] == 'None':
            return f'{output} = None'
        elif self.parameters['type'] in ('int', 'float', 'bool', 'int[]'):
            return f'{output} = {self.parameters["value"]}'
        elif self.parameters['type'] == 'str':
            str_val = self.parameters["value"]
            return f'{output} = "{str_val}"'
        elif self.parameters['type'] == 'Device':
            value = self.parameters['value']
            return f'{output} = flow.device("{value}")'
        else:
            raise RuntimeError(f'unsupported type of prim::Constant: {self.parameters["type"]}')

class PrimListConstruct(OneflowOperation):
    _ori_type_name = ['prim::ListConstruct']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = [{", ".join(inputs)}]'

class PrimListUnpack(OneflowOperation):
    _ori_type_name = ['prim::ListUnpack']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {inputs[0]}'

class PrimTupleConstruct(OneflowOperation):
    _ori_type_name = ['prim::TupleConstruct']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = ({", ".join(inputs)})'

class PrimTupleUnpack(OneflowOperation):
    _ori_type_name = ['prim::TupleUnpack']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # have single output here, because the following code uses index to access the unpacked values
        assert len(inputs) == 1
        return f'{output} = {inputs[0]}'

class PrimGetAttr(OneflowOperation):
    _ori_type_name = ['prim::GetAttr']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        if self.parameters['value'] is not None:
            return f"{output} = {self.parameters['value']}"
        else:
            return f"{output} = {self.parameters['input']}.{self.parameters['name']}"

class SimpleMember(OneflowOperation):
    _ori_type_name = ['prim::is_cuda', 'prim::data']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        member_name = self.type.split('::')[-1]
        return f'{output} = {inputs[0]}.{member_name}'

class AtenContiguous(OneflowOperation):
    _ori_type_name = ['aten::contiguous']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # defined in pytorch/c10/core/MemoryFormat.h
        assert inputs_value[1] in [0, 1, 2]
        return f'{output} = {inputs[0]}.contiguous(memory_format={mem_format[inputs_value[1]]})'

class AtenGetitem(OneflowOperation):
    _ori_type_name = ['aten::__getitem__']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        assert len(inputs) == 2
        return f'{output} = {inputs[0]}[{inputs[1]}]'

class AtenAppend(OneflowOperation):
    _ori_type_name = ['aten::append']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        assert len(inputs) == 2
        return f'_, {output} = {inputs[0]}.append({inputs[1]}), {inputs[0]}'

class MergedSlice(OneflowOperation):
    _ori_type_name = ['MergedSlice']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        if (len(inputs) - 1) % 4 == 0:
            slices = []
            dim = int((len(inputs) - 1) / 4)
            for i in range(dim):
                slices.append(f'{inputs[i*4+2]}:{inputs[i*4+3]}:{inputs[i*4+4]}')
            slice_str = ','.join(slices)
            return f'{output} = {inputs[0]}[{slice_str}]'
        elif len(inputs) == 4:
            # this case is for simple list
            return f'{output} = {inputs[0]}[{inputs[1]}:{inputs[2]}:{inputs[3]}]'
        else:
            raise RuntimeError('Unsupported slice pattern')

# the following Aten classes means these aten ops are not in flow.Tensor

class AtenBool(OneflowOperation):
    _ori_type_name = ['aten::Bool']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = bool({inputs[0]})'

class AtenNot(OneflowOperation):
    _ori_type_name = ['aten::__not__']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = not {inputs[0]}'

class AtenCat(OneflowOperation):
    _ori_type_name = ['aten::cat']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        assert len(inputs) == 2
        return f'{output} = flow.cat({inputs[0]}, dim={inputs[1]})'


class AtenAvgpool2d(OneflowOperation):
    # NOTE: it is not included in the above aten ops for unkown reason
    _ori_type_name = ['aten::avg_pool2d']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = F.avg_pool2d({", ".join(inputs)})'