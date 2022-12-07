import pefile
import pandas as pd

# def characteristics(data) -> "list[str]":
#   names = []
#   for name, val in pefile.image_characteristics:
#     if data.FILE_HEADER.Characteristics & val:
#       names.append(name)
#   return names

# def characteristics_str(data) -> str:
#   return ", ".join([c.replace("IMAGE_FILE_", "") for c in characteristics(data)])


# def pe_format(data) -> str:
#   if data.OPTIONAL_HEADER.Magic == pefile.OPTIONAL_HEADER_MAGIC_PE:
#     return "PE32"
#   elif data.OPTIONAL_HEADER.Magic == pefile.OPTIONAL_HEADER_MAGIC_PE_PLUS:
#     return "PE32+"
#   return "unknown"

# def subsystem(data) -> str:
#   return pefile.SUBSYSTEM_TYPE[data.OPTIONAL_HEADER.Subsystem].replace(
#   "IMAGE_SUBSYSTEM_", ""
#   )

# def dll_characteristics(data) -> "list[str]":
#   names = []
#   for name, val in pefile.dll_characteristics:
#     if data.OPTIONAL_HEADER.DllCharacteristics & val:
#       names.append(name)
#   return names

# def dll_characteristics_str(data) -> str:
#   return ", ".join(
#     [
#       c.replace("IMAGE_DLLCHARACTERISTICS_", "").replace("IMAGE_LIBRARY_", "")
#       for c in dll_characteristics(data)
#     ]
#   )
  
def write(file):
  print(file)
  pe = pefile.PE(file)
  machine = float(pe.FILE_HEADER.Machine)
  num = float(pe.FILE_HEADER.NumberOfSections)
  char = float(pe.FILE_HEADER.Characteristics)
  PointerToSymbolTable = float(pe.FILE_HEADER.PointerToSymbolTable)
  NumberOfSymbols = float(pe.FILE_HEADER.NumberOfSymbols)
  SizeOfOptionalHeader = float(pe.FILE_HEADER.SizeOfOptionalHeader)
  Magic = float(pe.OPTIONAL_HEADER.Magic)
  MajorLinkerVersion = float(pe.OPTIONAL_HEADER.MajorLinkerVersion)
  MinorLinkerVersion = float(pe.OPTIONAL_HEADER.MinorLinkerVersion)
  SizeOfCode = float(pe.OPTIONAL_HEADER.SizeOfCode)
  SizeOfInitializedData = float(pe.OPTIONAL_HEADER.SizeOfInitializedData)
  SizeOfUninitializedData = float(pe.OPTIONAL_HEADER.SizeOfUninitializedData)
  AddressOfEntryPoint = float(pe.OPTIONAL_HEADER.AddressOfEntryPoint)
  BaseOfCode = float(pe.OPTIONAL_HEADER.BaseOfCode)
  ImageBase = float(pe.OPTIONAL_HEADER.ImageBase)
  SectionAlignment = float(pe.OPTIONAL_HEADER.SectionAlignment)
  FileAlignment = float(pe.OPTIONAL_HEADER.FileAlignment)
  MajorOperatingSystemVersion = float(pe.OPTIONAL_HEADER.MajorOperatingSystemVersion)
  MinorOperatingSystemVersion = float(pe.OPTIONAL_HEADER.MinorOperatingSystemVersion)
  MajorImageVersion = float(pe.OPTIONAL_HEADER.MajorImageVersion)
  MinorImageVersion = float(pe.OPTIONAL_HEADER.MinorImageVersion)
  MajorSubsystemVersion = float(pe.OPTIONAL_HEADER.MajorSubsystemVersion)
  MinorSubsystemVersion = float(pe.OPTIONAL_HEADER.MinorSubsystemVersion)
  Win32VersionValue = float(pe.OPTIONAL_HEADER.Reserved1)
  SizeOfImage = float(pe.OPTIONAL_HEADER.SizeOfImage)
  SizeOfHeaders = float(pe.OPTIONAL_HEADER.SizeOfHeaders)
  CheckSum = float(pe.OPTIONAL_HEADER.CheckSum)
  Subsystem = float(pe.OPTIONAL_HEADER.Subsystem)
  DllCharacteristics = float(pe.OPTIONAL_HEADER.DllCharacteristics)
  SizeOfStackReserve = float(pe.OPTIONAL_HEADER.SizeOfStackReserve)
  SizeOfStackCommit = float(pe.OPTIONAL_HEADER.SizeOfStackCommit)
  SizeOfHeapReserve = float(pe.OPTIONAL_HEADER.SizeOfHeapReserve)
  SizeOfHeapCommit = float(pe.OPTIONAL_HEADER.SizeOfHeapCommit)
  LoaderFlags = float(pe.OPTIONAL_HEADER.LoaderFlags)
  NumberOfRvaAndSizes = float(pe.OPTIONAL_HEADER.NumberOfRvaAndSizes)
  
  data = ({
    # header file
          'name' : [file],
          'Machine' :[machine],
          'NumberOfSections' :[num],
          'PointerToSymbolTable' :[PointerToSymbolTable],
          'NumberOfSymbols' : [NumberOfSymbols],
          'SizeOfOptionalHeader' : [SizeOfOptionalHeader],
          'Characteristics':[char],
          # optional header
          'Magic' : [Magic],
          'MajorLinkerVersion' : [MajorLinkerVersion],
          'MinorLinkerVersion' : [MinorLinkerVersion],
          'SizeOfCode' : [SizeOfCode],
          'SizeOfInitializedData' : [SizeOfInitializedData],
          'SizeOfUninitializedData' : [SizeOfUninitializedData],
          'AddressOfEntryPoint' : [AddressOfEntryPoint],
          'BaseOfCode' : [BaseOfCode],
          'ImageBase' : [ImageBase],
          'SectionAlignment' : [SectionAlignment],
          'FileAlignment' : [FileAlignment],
          'MajorOperatingSystemVersion' : [MajorOperatingSystemVersion],
          'MinorOperatingSystemVersion' : [MinorOperatingSystemVersion],
          'MajorImageVersion' : [MajorImageVersion],
          'MinorImageVersion' : [MinorImageVersion],
          'MajorSubsystemVersion' : [MajorSubsystemVersion],
          'MinorSubsystemVersion' : [MinorSubsystemVersion],
          'Win32VersionValue' : [Win32VersionValue],
          'SizeOfImage' : [SizeOfImage],
          'SizeOfHeaders' : [SizeOfHeaders],
          'CheckSum' : [CheckSum],
          'Subsystem' : [Subsystem],
          'DllCharacteristics' : [DllCharacteristics],
          'SizeOfStackReserve' : [SizeOfStackReserve],
          'SizeOfStackCommit' : [SizeOfStackCommit],
          'SizeOfHeapReserve' : [SizeOfHeapReserve],
          'SizeOfHeapCommit' : [SizeOfHeapCommit],
          'LoaderFlags' : [LoaderFlags],
          'NumberOfRvaAndSizes' : [NumberOfRvaAndSizes]
          
              })
  return data

# try:
#   file_data = concat.DataFrame(write("E:/sampel/vm-secu/dump/file.0xd606b5a6f8a0.0xd606b5a74b20.ImageSectionObject.services.exe.img"))
#   print(file_data)
#   file = concat.read_csv('data.csv')

#   i = 0
#   while i < len(file):
#     data = concat.DataFrame(write("E:/sampel/vm-secu/dump/"+file['nama'][i]))
#     # print(data)
#     file_data = file_data.append(data, ignore_index=True)
#     i += 1

# except Exception as e :
# # print(file_data)
#   print(i)
#   file_data.to_excel (r'1.xlsx', index = False, header=True)

file_data = pd.DataFrame(write("E:/sampel/Demo/dmp/pid.1004.0x1000000.dmp"))
print(file_data)
file = pd.read_csv('E:/sampel/Demo/dmp/Book1.csv')

# file.nama.str.contains('dll')
print(len(file))
#recursive funtion
# def repeat(i, file_data):
status = True
i = 0
while status :
  try:
    i = i
    while i < len(file):
      data = pd.DataFrame(write("E:/sampel/Demo/dmp/" + file['nama'][i]))  #jangan lupa garismiring di akhir
      file_data = file_data.append(data, ignore_index=True)
      print('ini data ke ',len(file_data))
      i += 1
      
    status= False

  except:
    i += 1
    print('error di', i)
    
file_data.to_csv(r'E:/sampel/Demo/dmp/informasifiledemo.csv', index = False, header=None)

    # repeat(i+1, file_data)
    
# repeat(0, file_data)
