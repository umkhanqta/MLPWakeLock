INSTRUCTION_CLASS_COLOR = {
	'nop': 0,
	'move': 1,
	'return': 2,
	'monitor': 3,
	'test': 4,
	'new': 5,
	'throw': 6,
	'jump': 7,
	'branch': 8,
	'arrayop': 9,
	'instanceop': 10,
	'staticop': 11,
	'invoke': 12,
	'unop': 13,
	'binop': 14
	} 

INSTRUCTION_CLASSES = [
	'nop',
	'move',
	'return',
	'monitor',
	'test',
	'new',
	'throw',
	'jump',
	'branch',
	'arrayop',
	'instanceop',
	'staticop',
	'invoke',
	'unop',
	'binop'
	]

INSTRUCTION_SET_COLOR = {
	'nop': 0,
	'move': 1,
	'move/from16': 1,
	'move/16': 1,
	'move-wide': 1,
	'move-wide/from16': 1,
	'move-wide/16': 1,
	'move-object': 1,
	'move-object/from16': 1,
	'move-object/16': 1,
	'move-result': 1,
	'move-result-wide': 1,
	'move-result-object': 1,
	'move-exception': 1,
	'return-void': 2,
	'return': 2,
	'return-wide': 2,
	'return-object': 2,
	'const/4': 1,
	'const/16': 1,
	'const': 1,
	'const/high16': 1,
	'const-wide/16': 1,
	'const-wide/32': 1,
	'const-wide': 1,
	'const-wide/high16': 1,
	'const-string': 1,
	'const-string/jumbo': 1,
	'const-class': 1,
	'monitor-enter': 3,
	'monitor-exit': 3,
	'check-cast': 4,
	'instance-of': 4,
	'array-length': 1,
	'new-instance': 5,
	'new-array': 5,
	'filled-new-array': 1,
	'filled-new-array/range': 1,
	'fill-array-data': 1,
	'fill-array-data-payload': 1,
	'throw': 6,
	'goto': 7,
	'goto/16': 7,
	'goto/32': 7,
	'packed-switch': 7,
	'packed-switch-payload': 7,
	'sparse-switch': 7,
	'sparse-switch-payload': 7,
	'cmpl-float': 7,
	'cmpg-float': 7,
	'cmpl-double': 7,
	'cmpg-double': 7,
	'cmp-long': 7,
	'if-eq': 8,
	'if-ne': 8,
	'if-lt': 8,
	'if-ge': 8,
	'if-gt': 8,
	'if-le': 8,
	'if-eqz': 8,
	'if-nez': 8,
	'if-ltz': 8,
	'if-gez': 8,
	'if-gtz': 8,
	'if-lez': 8,
	'aget': 9,
	'aget-wide': 9,
	'aget-object': 9,
	'aget-boolean': 9,
	'aget-byte': 9,
	'aget-char': 9,
	'aget-short': 9,
	'aput': 9,
	'aput-wide': 9,
	'aput-object': 9,
	'aput-boolean': 9,
	'aput-byte': 9,
	'aput-char': 9,
	'aput-short': 9,
	'iget': 10,
	'iget-wide': 10,
	'iget-object': 10,
	'iget-boolean': 10,
	'iget-byte': 10,
	'iget-char': 10,
	'iget-short': 10,
	'iput': 10,
	'iput-wide': 10,
	'iput-object': 10,
	'iput-boolean': 10,
	'iput-byte': 10,
	'iput-char': 10,
	'iput-short': 10,
	'sget': 11,
	'sget-wide': 11,
	'sget-object': 11,
	'sget-boolean': 11,
	'sget-byte': 11,
	'sget-char': 11,
	'sget-short': 11,
	'sput': 11,
	'sput-wide': 11,
	'sput-object': 11,
	'sput-boolean': 11,
	'sput-byte': 11,
	'sput-char': 11,
	'sput-short': 11,
	'invoke-virtual': 12,
	'invoke-super': 12,
	'invoke-direct': 12,
	'invoke-static': 12,
	'invoke-interface': 12,
	'invoke-virtual/range': 12,
	'invoke-super/range': 12,
	'invoke-direct/range': 12,
	'invoke-static/range': 12,
	'invoke-interface/range': 12,
	'neg-int': 13,
	'not-int': 13,
	'neg-long': 13,
	'not-long': 13,
	'neg-float': 13,
	'neg-double': 13,
	'int-to-long': 13,
	'int-to-float': 13,
	'int-to-double': 13,
	'long-to-int': 13,
	'long-to-float': 13,
	'long-to-double': 13,
	'float-to-int': 13,
	'float-to-long': 13,
	'float-to-double': 13,
	'double-to-int': 13,
	'double-to-long': 13,
	'double-to-float': 13,
	'int-to-byte': 13,
	'int-to-char': 13,
	'int-to-short': 13,
	'add-int': 14,
	'sub-int': 14,
	'mul-int': 14,
	'div-int': 14,
	'rem-int': 14,
	'and-int': 14,
	'or-int': 14,
	'xor-int': 14,
	'shl-int': 14,
	'shr-int': 14,
	'ushr-int': 14,
	'add-long': 14,
	'sub-long': 14,
	'mul-long': 14,
	'div-long': 14,
	'rem-long': 14,
	'and-long': 14,
	'or-long': 14,
	'xor-long': 14,
	'shl-long': 14,
	'shr-long': 14,
	'ushr-long': 14,
	'add-float': 14,
	'sub-float': 14,
	'mul-float': 14,
	'div-float': 14,
	'rem-float': 14,
	'add-double': 14,
	'sub-double': 14,
	'mul-double': 14,
	'div-double': 14,
	'rem-double': 14,
	'add-int/2addr': 14,
	'sub-int/2addr': 14,
	'mul-int/2addr': 14,
	'div-int/2addr': 14,
	'rem-int/2addr': 14,
	'and-int/2addr': 14,
	'or-int/2addr': 14,
	'xor-int/2addr': 14,
	'shl-int/2addr': 14,
	'shr-int/2addr': 14,
	'ushr-int/2addr': 14,
	'add-long/2addr': 14,
	'sub-long/2addr': 14,
	'mul-long/2addr': 14,
	'div-long/2addr': 14,
	'rem-long/2addr': 14,
	'and-long/2addr': 14,
	'or-long/2addr': 14,
	'xor-long/2addr': 14,
	'shl-long/2addr': 14,
	'shr-long/2addr': 14,
	'ushr-long/2addr': 14,
	'add-float/2addr': 14,
	'sub-float/2addr': 14,
	'mul-float/2addr': 14,
	'div-float/2addr': 14,
	'rem-float/2addr': 14,
	'add-double/2addr': 14,
	'sub-double/2addr': 14,
	'mul-double/2addr': 14,
	'div-double/2addr': 14,
	'rem-double/2addr': 14,
	'add-int/lit16': 14,
	'rsub-int': 14,
	'mul-int/lit16': 14,
	'div-int/lit16': 14,
	'rem-int/lit16': 14,
	'and-int/lit16': 14,
	'or-int/lit16': 14,
	'xor-int/lit16': 14,
	'add-int/lit8': 14,
	'rsub-int/lit8': 14,
	'mul-int/lit8': 14,
	'div-int/lit8': 14,
	'rem-int/lit8': 14,
	'and-int/lit8': 14,
	'or-int/lit8': 14,
	'xor-int/lit8': 14,
	'shl-int/lit8': 14,
	'shr-int/lit8': 14,
	'ushr-int/lit8': 14
	}


"""
INSTRUCTION SET FROM ANDROGUARD

INSTRUCTION_SET = {
    'nop':                    nop,
    'move':                   move,
    'move/from16':            movefrom16,
    'move/16':                move16,
    'move-wide':              movewide,
    'move-wide/from16':       movewidefrom16,
    'move-wide/16':           movewide16,
    'move-object':            moveobject,
    'move-object/from16':     moveobjectfrom16,
    'move-object/16':         moveobject16,
    'move-result':            moveresult,
    'move-result-wide':       moveresultwide,
    'move-result-object':     moveresultobject,
    'move-exception':         moveexception,
    'return-void':            returnvoid,
    'return':                 return_reg,
    'return-wide':            returnwide,
    'return-object':          returnobject,
    'const/4':                const4,
    'const/16':               const16,
    'const':                  const,
    'const/high16':           consthigh16,
    'const-wide/16':          constwide16,
    'const-wide/32':          constwide32,
    'const-wide':             constwide,
    'const-wide/high16':      constwidehigh16,
    'const-string':           conststring,
    'const-string/jumbo':     conststringjumbo,
    'const-class':            constclass,
    'monitor-enter':          monitorenter,
    'monitor-exit':           monitorexit,
    'check-cast':             checkcast,
    'instance-of':            instanceof,
    'array-length':           arraylength,
    'new-instance':           newinstance,
    'new-array':              newarray,
    'filled-new-array':       fillednewarray,
    'filled-new-array/range': fillednewarrayrange,
    'fill-array-data':        fillarraydata,
    'fill-array-data-payload': fillarraydatapayload,
    'throw':                  throw,
    'goto':                   goto,
    'goto/16':                goto16,
    'goto/32':                goto32,
    'packed-switch':          packedswitch,
    'sparse-switch':          sparseswitch,
    'cmpl-float':             cmplfloat,
    'cmpg-float':             cmpgfloat,
    'cmpl-double':            cmpldouble,
    'cmpg-double':            cmpgdouble,
    'cmp-long':               cmplong,
    'if-eq':                  ifeq,
    'if-ne':                  ifne,
    'if-lt':                  iflt,
    'if-ge':                  ifge,
    'if-gt':                  ifgt,
    'if-le':                  ifle,
    'if-eqz':                 ifeqz,
    'if-nez':                 ifnez,
    'if-ltz':                 ifltz,
    'if-gez':                 ifgez,
    'if-gtz':                 ifgtz,
    'if-lez':                 iflez,
    'aget':                   aget,
    'aget-wide':              agetwide,
    'aget-object':            agetobject,
    'aget-boolean':           agetboolean,
    'aget-byte':              agetbyte,
    'aget-char':              agetchar,
    'aget-short':             agetshort,
    'aput':                   aput,
    'aput-wide':              aputwide,
    'aput-object':            aputobject,
    'aput-boolean':           aputboolean,
    'aput-byte':              aputbyte,
    'aput-char':              aputchar,
    'aput-short':             aputshort,
    'iget':                   iget,
    'iget-wide':              igetwide,
    'iget-object':            igetobject,
    'iget-boolean':           igetboolean,
    'iget-byte':              igetbyte,
    'iget-char':              igetchar,
    'iget-short':             igetshort,
    'iput':                   iput,
    'iput-wide':              iputwide,
    'iput-object':            iputobject,
    'iput-boolean':           iputboolean,
    'iput-byte':              iputbyte,
    'iput-char':              iputchar,
    'iput-short':             iputshort,
    'sget':                   sget,
    'sget-wide':              sgetwide,
    'sget-object':            sgetobject,
    'sget-boolean':           sgetboolean,
    'sget-byte':              sgetbyte,
    'sget-char':              sgetchar,
    'sget-short':             sgetshort,
    'sput':                   sput,
    'sput-wide':              sputwide,
    'sput-object':            sputobject,
    'sput-boolean':           sputboolean,
    'sput-byte':              sputbyte,
    'sput-char':              sputchar,
    'sput-short':             sputshort,
    'invoke-virtual':         invokevirtual,
    'invoke-super':           invokesuper,
    'invoke-direct':          invokedirect,
    'invoke-static':          invokestatic,
    'invoke-interface':       invokeinterface,
    'invoke-virtual/range':   invokevirtualrange,
    'invoke-super/range':     invokesuperrange,
    'invoke-direct/range':    invokedirectrange,
    'invoke-static/range':    invokestaticrange,
    'invoke-interface/range': invokeinterfacerange,
    'neg-int':                negint,
    'not-int':                notint,
    'neg-long':               neglong,
    'not-long':               notlong,
    'neg-float':              negfloat,
    'neg-double':             negdouble,
    'int-to-long':            inttolong,
    'int-to-float':           inttofloat,
    'int-to-double':          inttodouble,
    'long-to-int':            longtoint,
    'long-to-float':          longtofloat,
    'long-to-double':         longtodouble,
    'float-to-int':           floattoint,
    'float-to-long':          floattolong,
    'float-to-double':        floattodouble,
    'double-to-int':          doubletoint,
    'double-to-long':         doubletolong,
    'double-to-float':        doubletofloat,
    'int-to-byte':            inttobyte,
    'int-to-char':            inttochar,
    'int-to-short':           inttoshort,
    'add-int':                addint,
    'sub-int':                subint,
    'mul-int':                mulint,
    'div-int':                divint,
    'rem-int':                remint,
    'and-int':                andint,
    'or-int':                 orint,
    'xor-int':                xorint,
    'shl-int':                shlint,
    'shr-int':                shrint,
    'ushr-int':               ushrint,
    'add-long':               addlong,
    'sub-long':               sublong,
    'mul-long':               mullong,
    'div-long':               divlong,
    'rem-long':               remlong,
    'and-long':               andlong,
    'or-long':                orlong,
    'xor-long':               xorlong,
    'shl-long':               shllong,
    'shr-long':               shrlong,
    'ushr-long':              ushrlong,
    'add-float':              addfloat,
    'sub-float':              subfloat,
    'mul-float':              mulfloat,
    'div-float':              divfloat,
    'rem-float':              remfloat,
    'add-double':             adddouble,
    'sub-double':             subdouble,
    'mul-double':             muldouble,
    'div-double':             divdouble,
    'rem-double':             remdouble,
    'add-int/2addr':          addint2addr,
    'sub-int/2addr':          subint2addr,
    'mul-int/2addr':          mulint2addr,
    'div-int/2addr':          divint2addr,
    'rem-int/2addr':          remint2addr,
    'and-int/2addr':          andint2addr,
    'or-int/2addr':           orint2addr,
    'xor-int/2addr':          xorint2addr,
    'shl-int/2addr':          shlint2addr,
    'shr-int/2addr':          shrint2addr,
    'ushr-int/2addr':         ushrint2addr,
    'add-long/2addr':         addlong2addr,
    'sub-long/2addr':         sublong2addr,
    'mul-long/2addr':         mullong2addr,
    'div-long/2addr':         divlong2addr,
    'rem-long/2addr':         remlong2addr,
    'and-long/2addr':         andlong2addr,
    'or-long/2addr':          orlong2addr,
    'xor-long/2addr':         xorlong2addr,
    'shl-long/2addr':         shllong2addr,
    'shr-long/2addr':         shrlong2addr,
    'ushr-long/2addr':        ushrlong2addr,
    'add-float/2addr':        addfloat2addr,
    'sub-float/2addr':        subfloat2addr,
    'mul-float/2addr':        mulfloat2addr,
    'div-float/2addr':        divfloat2addr,
    'rem-float/2addr':        remfloat2addr,
    'add-double/2addr':       adddouble2addr,
    'sub-double/2addr':       subdouble2addr,
    'mul-double/2addr':       muldouble2addr,
    'div-double/2addr':       divdouble2addr,
    'rem-double/2addr':       remdouble2addr,
    'add-int/lit16':          addintlit16,
    'rsub-int':               rsubint,
    'mul-int/lit16':          mulintlit16,
    'div-int/lit16':          divintlit16,
    'rem-int/lit16':          remintlit16,
    'and-int/lit16':          andintlit16,
    'or-int/lit16':           orintlit16,
    'xor-int/lit16':          xorintlit16,
    'add-int/lit8':           addintlit8,
    'rsub-int/lit8':          rsubintlit8,
    'mul-int/lit8':           mulintlit8,
    'div-int/lit8':           divintlit8,
    'rem-int/lit8':           remintlit8,
    'and-int/lit8':           andintlit8,
    'or-int/lit8':            orintlit8,
    'xor-int/lit8':           xorintlit8,
    'shl-int/lit8':           shlintlit8,
    'shr-int/lit8':           shrintlit8,
    'ushr-int/lit8':          ushrintlit8,
}
"""
