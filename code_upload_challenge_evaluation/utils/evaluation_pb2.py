# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: evaluation.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='evaluation.proto',
  package='evaluation',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x10\x65valuation.proto\x12\nevaluation\"#\n\x07Package\x12\x18\n\x10SerializedEntity\x18\x01 \x01(\x0c\x32\x8f\x01\n\x0b\x45nvironment\x12>\n\x10get_action_space\x12\x13.evaluation.Package\x1a\x13.evaluation.Package\"\x00\x12@\n\x12\x61\x63t_on_environment\x12\x13.evaluation.Package\x1a\x13.evaluation.Package\"\x00\x62\x06proto3')
)




_PACKAGE = _descriptor.Descriptor(
  name='Package',
  full_name='evaluation.Package',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='SerializedEntity', full_name='evaluation.Package.SerializedEntity', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=32,
  serialized_end=67,
)

DESCRIPTOR.message_types_by_name['Package'] = _PACKAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Package = _reflection.GeneratedProtocolMessageType('Package', (_message.Message,), {
  'DESCRIPTOR' : _PACKAGE,
  '__module__' : 'evaluation_pb2'
  # @@protoc_insertion_point(class_scope:evaluation.Package)
  })
_sym_db.RegisterMessage(Package)



_ENVIRONMENT = _descriptor.ServiceDescriptor(
  name='Environment',
  full_name='evaluation.Environment',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=70,
  serialized_end=213,
  methods=[
  _descriptor.MethodDescriptor(
    name='get_action_space',
    full_name='evaluation.Environment.get_action_space',
    index=0,
    containing_service=None,
    input_type=_PACKAGE,
    output_type=_PACKAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='act_on_environment',
    full_name='evaluation.Environment.act_on_environment',
    index=1,
    containing_service=None,
    input_type=_PACKAGE,
    output_type=_PACKAGE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_ENVIRONMENT)

DESCRIPTOR.services_by_name['Environment'] = _ENVIRONMENT

# @@protoc_insertion_point(module_scope)
