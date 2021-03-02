# Web UI for configuration of the CMSIS-DSP Build
#
# How to install
# pip install streamlit
#
# How to use
# streamlit run cmsisconfig.py
#
import streamlit as st
import textwrap
import re


st.set_page_config(page_title="CMSIS-DSP Configuration",layout="wide" )

# Options requiring a special management
NOTSTANDARD=["allTables","allInterpolations","allFFTs","Float16"]

HELIUM=False

config={}

config["allTables"] = True
config["allFFTs"] = True
config["allInterpolations"] = True
config["MVEI"]=False
config["MVEF"]=False
config["NEON"]=False
config["HELIUM"]=False
config["HELIUMEXPERIMENTAL"]=False
config["Float16"]=True
config["HOST"]=False

config["COS_F32"]=False
config["COS_Q31"]=False
config["COS_Q15"]=False
config["SIN_F32"]=False
config["SIN_Q31"]=False
config["SIN_Q15"]=False
config["SIN_COS_F32"]=False
config["SIN_COS_Q31"]=False
config["LMS_NORM_Q31"]=False
config["LMS_NORM_Q15"]=False
config["CMPLX_MAG_Q31"]=False
config["CMPLX_MAG_Q15"]=False

config["BASICMATH"]=True  
config["COMPLEXMATH"]=True
config["CONTROLLER"]=True    
config["FASTMATH"]=True      
config["FILTERING"]=True     
config["MATRIX"]=True       
config["STATISTICS"]=True    
config["SUPPORT"]=True       
config["TRANSFORM"]=True   
config["SVM"]=True           
config["BAYES"]=True        
config["DISTANCE"]=True     
config["INTERPOLATION"]=True
config["QUATERNIONMATH"]=True

config["LOOPUNROLL"]=True
config["ROUNDING"]=False
config["MATRIXCHECK"]=False
config["AUTOVECTORIZE"] = False

realname={}
realname["COS_F32"]="ARM_COS_F32"
realname["COS_Q31"]="ARM_COS_Q31"
realname["COS_Q15"]="ARM_COS_Q15"
realname["SIN_F32"]="ARM_SIN_F32"
realname["SIN_Q31"]="ARM_SIN_Q31"
realname["SIN_Q15"]="ARM_SIN_Q15"
realname["SIN_COS_F32"]="ARM_SIN_COS_F32"
realname["SIN_COS_Q31"]="ARM_SIN_COS_Q31"
realname["LMS_NORM_Q31"]="ARM_LMS_NORM_Q31"
realname["LMS_NORM_Q15"]="ARM_LMS_NORM_Q15"
realname["CMPLX_MAG_Q31"]="ARM_CMPLX_MAG_Q31"
realname["CMPLX_MAG_Q15"]="ARM_CMPLX_MAG_Q15"

defaulton={}
defaulton["LOOPUNROLL"]=True 
defaulton["BASICMATH"]=True  
defaulton["COMPLEXMATH"]=True
defaulton["CONTROLLER"]=True   
defaulton["FASTMATH"]=True     
defaulton["FILTERING"]=True    
defaulton["MATRIX"]=True       
defaulton["STATISTICS"]=True   
defaulton["SUPPORT"]=True      
defaulton["TRANSFORM"]=True   
defaulton["SVM"]=True          
defaulton["BAYES"]=True        
defaulton["DISTANCE"]=True     
defaulton["INTERPOLATION"]=True
defaulton["QUATERNIONMATH"]=True


CFFTSIZE=[16,32,64,128,256,512,1024,2048,4096]
CFFTDATATYPE=['F64','F32','F16','Q31','Q15']

RFFTFASTSIZE=[32,64,128,256,512,1024,2048,4096]
RFFTFASTDATATYPE=['F64','F32','F16']

RFFTSIZE=[32,64,128,256,512,1024,2048,4096,8192]
RFFTDATATYPE=['F32','Q31','Q15']

DCTSIZE=[128,512,2048,8192]
DCTDATATYPE=['F32','Q31','Q15']

def joinit(iterable, delimiter):
    # Intersperse a delimiter between element of a list
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

def options(l):
    return("".join(joinit(l," ")))

def computeCmakeOptions(config):
    global defaulton
    cmake={}
    if not config["allTables"]:
       cmake["CONFIGTABLE"]=True
       if config["allInterpolations"]:
          cmake["ALLFAST"]=True
       if config["allFFTs"]:
          cmake["ALLFFT"]=True
    if config["Float16"]:
       cmake["FLOAT16"]=True
    else:
       cmake["DISABLEFLOAT16"]=True

    for c in config:
        if not (c in NOTSTANDARD):
           if c in defaulton:
                if not config[c]:
                   if c in realname:
                      cmake[realname[c]]=False
                   else:
                      cmake[c]=False
           else:
                if config[c]:
                   if c in realname:
                      cmake[realname[c]]=True
                   else:
                      cmake[c]=True
    return cmake 

def removeDuplicates(l):
  return list(dict.fromkeys(l))

def genCMakeOptions(config):
    r=[]
    cmake = computeCmakeOptions(config)
    for c in cmake:
        if cmake[c]:
           r.append("-D%s=ON" % c)
        else:
           r.append("-D%s=OFF" % c)
    return(removeDuplicates(r),cmake)

def test(cmake,s):
    global defaulton
    if s in defaulton and not (s in cmake):
       return True 
    return(s in cmake and cmake[s])

def cfftCF32Config(cmake,size):
    result=[]
    if test(cmake,"CFFT_F32_%d" % size):
       a="-DARM_TABLE_TWIDDLECOEF_F32_%d" % size
       if HELIUM:
          b = "-DARM_TABLE_BITREVIDX_FXT_%d" % size
       else:
          b = "-DARM_TABLE_BITREVIDX_FLT_%d" % size 
       result=[a,b]
    return(result)

def cfftCF16Config(cmake,size):
    result=[]
    if test(cmake,"CFFT_F16_%d" % size):
       result =["-DARM_TABLE_TWIDDLECOEF_F16_%d" % size]
       result.append("-DARM_TABLE_BITREVIDX_FXT_%d" % size)
       result.append("-DARM_TABLE_BITREVIDX_FLT_%d" % size)
    return(result)

def cfftCF64Config(cmake,size):
    result=[]
    if test(cmake,"CFFT_F64_%d" % size):
       result =["-DARM_TABLE_TWIDDLECOEF_F64_%d" % size]
       result.append("-DARM_TABLE_BITREVIDX_FLT64_%d" % size)
    return(result)


def cfftCFixedConfig(cmake,dt,size):
    result=[]
    if test(cmake,"CFFT_%s_%d" % (dt,size)):
       a="-DARM_TABLE_TWIDDLECOEF_%s_%d" % (dt,size)
       b = "-DARM_TABLE_BITREVIDX_FXT_%d" % size
       result=[a,b]
    return(result)

def crfftFastCF64Config(cmake,size):
    result=[]
    s1 = size >> 1
    if test(cmake,"RFFT_FAST_F64_%d" % size):
       result =[]
       result.append("-DARM_TABLE_TWIDDLECOEF_F64_%d" % s1)
       result.append("-DARM_TABLE_BITREVIDX_FLT64_%d" % s1)
       result.append("-DARM_TABLE_TWIDDLECOEF_RFFT_F64_%d" % size)
       result.append("-DARM_TABLE_TWIDDLECOEF_F64_%d" % s1)
       
    return(result)

def crfftFastCF32Config(cmake,size):
    result=[]
    s1 = size >> 1
    if test(cmake,"RFFT_FAST_F32_%d" % size):
       result =[]
       result.append("-DARM_TABLE_TWIDDLECOEF_F32_%d" % s1)
       result.append("-DARM_TABLE_BITREVIDX_FLT_%d" % s1)
       result.append("-DARM_TABLE_TWIDDLECOEF_RFFT_F32_%d" % size)
       
    return(result)

def crfftFastCF16Config(cmake,size):
    result=[]
    s1 = size >> 1
    if test(cmake,"RFFT_FAST_F16_%d" % size):
       result =[]
       result.append("-DARM_TABLE_TWIDDLECOEF_F16_%d" % s1)
       result.append("-DARM_TABLE_BITREVIDX_FLT_%d" % s1)
       result.append("-DARM_TABLE_BITREVIDX_FXT_%d" % s1)
       result.append("-DARM_TABLE_TWIDDLECOEF_RFFT_F16_%d" % size)
       
    return(result)

# Deprecated RFFT used in DCT
def crfftF32Config(cmake,size):
    result=[]
    s1 = size >> 1
    if test(cmake,"RFFT_FAST_F16_%d" % size):
       result =[]
       result.append("-DARM_TABLE_REALCOEF_F32")
       result.append("-ARM_TABLE_BITREV_%d" % s1)
       result.append("-ARM_TABLE_TWIDDLECOEF_F32_%d" % s1)
       
    return(result)


def crfftFixedConfig(cmake,dt,size):
    result=[]
    s1 = size >> 1
    if test(cmake,"RFFT_%s_%d" % (dt,size)):
       result =[]
       result.append("-DARM_TABLE_REALCOEF_%s" % dt)
       result.append("-DARM_TABLE_TWIDDLECOEF_%s_%d" % (dt,s1))
       result.append("-DARM_TABLE_BITREVIDX_FXT_%d" % s1)
       
    return(result)


def dctConfig(cmake,dt,size):
    result=[]
    if test(cmake,"DCT4_%s_%d" % (dt,size)):
       result =[]
       result.append("-DARM_TABLE_DCT4_%s_%d" % (dt,size))
       result.append("-DARM_TABLE_REALCOEF_F32")
       result.append("-DARM_TABLE_BITREV_1024" )
       result.append("-DARM_TABLE_TWIDDLECOEF_%s_4096" % dt)
       
    return(result)

# Convert cmake options to make flags
def interpretCmakeOptions(cmake):
    r=[]
    if test(cmake,"CONFIGTABLE"):
       r.append("-DARM_DSP_CONFIG_TABLES")
       # In Make configuration we build all modules.
       # So the code for FFT and FAST maths may be included
       # so we allow the table to be included if they are needed.
       r.append("-DARM_FAST_ALLOW_TABLES")
       r.append("-DARM_FFT_ALLOW_TABLES")
       for size in CFFTSIZE:
           r += cfftCF32Config(cmake,size)
           r += cfftCF16Config(cmake,size)
           r += cfftCF64Config(cmake,size)
           r += cfftCFixedConfig(cmake,"Q31",size)
           r += cfftCFixedConfig(cmake,"Q15",size)

       for size in RFFTFASTSIZE:
          r += crfftFastCF64Config(cmake,size)
          r += crfftFastCF32Config(cmake,size)
          r += crfftFastCF16Config(cmake,size)

       for size in RFFTSIZE:
          r += crfftFixedConfig(cmake,"F32",size)
          r += crfftFixedConfig(cmake,"Q31",size)
          r += crfftFixedConfig(cmake,"Q15",size)

       for size in DCTSIZE:
          r += dctConfig(cmake,"F32",size)
          r += dctConfig(cmake,"Q31",size)
          r += dctConfig(cmake,"Q15",size)

        
    
    if test(cmake,"ALLFAST"):
       r.append("-DARM_ALL_FAST_TABLES")
    if test(cmake,"ALLFFT"):
       r.append("-DARM_ALL_FFT_TABLES")

    if test(cmake,"LOOPUNROLL"):
       r.append("-DARM_MATH_LOOPUNROLL")
    if test(cmake,"ROUNDING"):
       r.append("-DARM_MATH_ROUNDING")
    if test(cmake,"MATRIXCHECK"):
       r.append("-DARM_MATH_MATRIX_CHECK")
    if test(cmake,"AUTOVECTORIZE"):
       r.append("-DARM_MATH_AUTOVECTORIZE")
    if test(cmake,"DISABLEFLOAT16"):
       r.append("-DDISABLEFLOAT16")
    if test(cmake,"NEON"):
       r.append("-DARM_MATH_NEON")
       r.append("-DARM_MATH_NEON_EXPERIMENTAL")
    if test(cmake,"HOST"):
        r.append("-D__GNUC_PYTHON__")
    
    if test(cmake,"ARM_COS_F32"):
        r.append("-DARM_TABLE_SIN_F32")
    if test(cmake,"ARM_COS_Q31"):
        r.append("-DARM_TABLE_SIN_Q31")
    if test(cmake,"ARM_COS_Q15"):
        r.append("-DARM_TABLE_SIN_Q15")

    if test(cmake,"ARM_SIN_F32"):
        r.append("-DARM_TABLE_SIN_F32")
    if test(cmake,"ARM_SIN_Q31"):
        r.append("-DARM_TABLE_SIN_Q31")
    if test(cmake,"ARM_SIN_Q15"):
        r.append("-DARM_TABLE_SIN_Q15")

    if test(cmake,"ARM_SIN_COS_F32"):
        r.append("-DARM_TABLE_SIN_F32")
    if test(cmake,"ARM_SIN_COS_Q31"):
        r.append("-DARM_TABLE_SIN_Q31")

    if test(cmake,"ARM_LMS_NORM_Q31"):
        r.append("-DARM_TABLE_RECIP_Q31")

    if test(cmake,"ARM_LMS_NORM_Q15"):
        r.append("-DARM_TABLE_RECIP_Q15")

    if test(cmake,"ARM_CMPLX_MAG_Q31"):
        r.append("-DARM_TABLE_FAST_SQRT_Q31_MVE")

    if test(cmake,"ARM_CMPLX_MAG_Q15"):
        r.append("-DARM_TABLE_FAST_SQRT_Q15_MVE")

    if test(cmake,"MVEI"):
       r.append("-DARM_MATH_MVEI")

    if test(cmake,"MVEF"):
       r.append("-DARM_MATH_MVEF")

    if test(cmake,"HELIUMEXPERIMENTAL"):
       r.append("-DARM_MATH_HELIUM_EXPERIMENTAL")

    if test(cmake,"HELIUM") or test(cmake,"MVEF") or test(cmake,"MVEI"):
       r.append("-IPrivateInclude")

    if test(cmake,"NEON") or test(cmake,"NEONEXPERIMENTAL"):
       r.append("-IComputeLibrary/Include")

    return (removeDuplicates(r))

def genMakeOptions(config):
    cmake = computeCmakeOptions(config)
    r=interpretCmakeOptions(cmake)
    return(r,cmake)


def check(config,s,name=None,comment=None):
    if comment is not None:
       st.sidebar.text(comment)
    if name is None:
       config[s]=st.sidebar.checkbox(s,value=config[s])
    else:
       config[s]=st.sidebar.checkbox(name,value=config[s])
    return(config[s])

def genconfig(config,transform,sizes,datatypes):
    global realname
    for size in sizes:
        for dt in datatypes:
            s="%s_%s_%s" % (transform,dt,size)
            config[s] = False
            realname[s] = s

def hasDCTF32(config):
    result=False
    for size in DCTSIZE:
            s="DCT4_F32_%s" % size
            if config[s]:
               result = True
    return(result)

def multiselect(config,name,options):
    default=[]
    for r in options:
        if config[r]:
            default.append(r)
    result=st.sidebar.multiselect(name,options,default=default)
    for r in options:
        config[r] = False
    for r in result:
        config[r] = True

def genui(config,transform,sizes,datatypes):
    keepF32 = True
    # RFFT F32 is deprecated and needed only for DCT4
    if transform == "RFFT":
       keepF32 = hasDCTF32(config)
    selected=st.sidebar.multiselect("Sizes",sizes)
    for size in selected:
        options=[]
        for dt in datatypes:
            if dt != "F32" or keepF32:
               s="%s_%s_%s" % (transform,dt,size)
               options.append(s)
        multiselect(config,"Nb = %d" % size,options)


def configMake(config):
    st.sidebar.header('Table Configuration')
    st.sidebar.info("Several options to include only the tables needed in an app and minimize code size.")
    if not check(config,"allTables","All tables included"):

        if not check(config,"allFFTs","All FFT tables included"):
           st.sidebar.markdown("#### CFFT")
           genui(config,"CFFT",CFFTSIZE,CFFTDATATYPE)

           st.sidebar.info("Following transforms are using the CFFT. You need to enable the needed CFFTs above.")

           st.sidebar.markdown("#### RFFT FAST")
           genui(config,"RFFT_FAST",RFFTFASTSIZE,RFFTFASTDATATYPE)
           st.sidebar.markdown("#### DCT4")
           genui(config,"DCT4",DCTSIZE,DCTDATATYPE)
           st.sidebar.markdown("#### RFFT")
           genui(config,"RFFT",RFFTSIZE,RFFTDATATYPE)
           



        if not check(config,"allInterpolations",'All interpolation tables included'):
           selected=st.sidebar.multiselect("Functions",["Cosine","Sine","SineCosine","Normalized LMS"])
           for s in selected:
               if s == "Cosine":
                  multiselect(config,"Cosine",["COS_F32","COS_Q31","COS_Q15"])
               if s == "Sine":
                  multiselect(config,"Sine",["SIN_F32","SIN_Q31","SIN_Q15"])
               if s == "SineCosine":
                  multiselect(config,"SineCosine",["SIN_COS_F32","SIN_COS_Q31"])
               if s == "Normalized LMS":
                  multiselect(config,"Normalized LMS",["LMS_NORM_Q31","LMS_NORM_Q15"])

           if config["MVEI"]:
              st.sidebar.markdown("#### Complex Magnitude")
              multiselect(config,"Complex Magnitude",["CMPLX_MAG_Q31","CMPLX_MAG_Q15"])



def configCMake(config):
    multiselect(config,"Folders",["BASICMATH",     
                       "COMPLEXMATH",   
                       "CONTROLLER",    
                       "FASTMATH",      
                       "FILTERING",     
                       "MATRIX",        
                       "STATISTICS",    
                       "SUPPORT",       
                       "TRANSFORM",     
                       "SVM",           
                       "BAYES",         
                       "DISTANCE",      
                       "INTERPOLATION","QUATERNIONMATH"])  
    configMake(config)

genconfig(config,"CFFT",CFFTSIZE,CFFTDATATYPE)
genconfig(config,"RFFT_FAST",RFFTFASTSIZE,RFFTFASTDATATYPE)
genconfig(config,"RFFT",RFFTSIZE,RFFTDATATYPE)
genconfig(config,"DCT4",DCTSIZE,DCTDATATYPE)

st.title('CMSIS-DSP Configuration')

st.warning("It is a work in progress. Only a small subset of the combinations has been tested.")

st.sidebar.header('Feature Configuration')
st.sidebar.info("To build on host. All features will be enabled.")
forHost=check(config,"HOST")

if not forHost:
   st.sidebar.info("Enable or disable float16 support")
   check(config,"Float16")

   st.sidebar.info("Some configurations for the CMSIS-DSP code.")
   check(config,"LOOPUNROLL")
   st.sidebar.text("Decrease performances when selected:")
   check(config,"ROUNDING")
   check(config,"MATRIXCHECK")
   
   st.sidebar.header('Vector extensions')
   st.sidebar.info("Enable vector code. It is not automatic for Neon. Use of Helium will enable new options to select some interpolation tables.")
   archi=st.sidebar.selectbox("Vector",('None','Helium','Neon'))
   if archi == 'Neon':
       config["NEON"]=True
   if archi == 'Helium':
      multiselect(config,"MVE configuration",["MVEI","MVEF"])
      HELIUM=True
      st.sidebar.info("When checked some experimental versions will be enabled and may be less performant than scalar version depending on the architecture.")
      check(config,"HELIUMEXPERIMENTAL")
   if archi != 'None':
       st.sidebar.info("When autovectorization is on, pure C code will be compiled. The version with C intrinsics won't be compiled.")
       check(config,"AUTOVECTORIZE")



st.sidebar.header('Build Method')

st.sidebar.info("With cmake, some folders can be removed from the build.")
selected=st.sidebar.selectbox('Select', ("Make","Cmake"),index=1)



if selected == "Make":
    if not forHost:
       configMake(config)
    result,cmake=genMakeOptions(config)
else:
    if not forHost:
       configCMake(config)
    result,cmake=genCMakeOptions(config)

st.header('Build options for %s command line' % selected)

if selected == "Make":
    if test(cmake,"FLOAT16"):
            st.info("Float16 is selected. You may need to pass compiler specific options for the compiler to recognize the float16 type.")

mode=st.selectbox("Mode",["txt","MDK","sh","bat"])

if mode=="txt":
   st.code(textwrap.fill(options(result)))

if mode=="MDK":
   opts=options(result)
   includes=""
   maybeincludes=re.findall(r'\-I([^\s]+)',opts)
   # Managed in MDK pack file
   #if maybeincludes:
   #   includes = maybeincludes
   #   st.text("Following include directories must be added")
   #   st.code(includes)
   opts=re.sub(r'\-D','',opts)
   opts=re.sub(r'\-I[^\s]+','',opts)
   st.text("MDK Preprocessor Symbols ")
   st.code(opts)


if mode=="sh":
   lines=options(result).split() 
   txt=""
   for l in lines:
       txt += " %s \\\n" % l
   txt += "\n"
   st.code(txt)

if mode=="bat":
   lines=options(result).split() 
   txt=""
   for l in lines:
       txt += " %s ^\n" % l
   txt += "\n"
   st.code(txt)

