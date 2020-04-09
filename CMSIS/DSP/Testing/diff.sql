.headers ON
/*

Select the core to be used as reference. Only last day of measurements is used.

*/
CREATE TEMP VIEW if not exists refCore AS select *
  from Unary
  where coreid=5 AND DATE BETWEEN datetime('now','localtime','-23 hours') AND datetime('now', 'localtime');
  ;

/*

Select the cores to be benchmarked compared with the reference. Only last day of measurements is used.

*/
CREATE TEMP VIEW if not exists  otherCores AS select *
  from Unary
  where coreid != 5 AND DATE BETWEEN datetime('now','localtime','-23 hours') AND datetime('now', 'localtime');
  ;

/*

Using regression database, compute the ratio using max cycles 
and max degree regression coefficient.

Change name of columns for result

USING(ID,categoryid,NAME) : would have to be extended with any parameter defining the regression
formula.
For instamce, for FFT, if ifft is an external parameter then ifft flag should be
here.

Here we assume ref and others are generated with same settings and currentConfig.csv
If not the case, the parameters which may be different (like LOOPUNROLL, OPTIMIZED...) 
should be added here so that we join the ref and other cores on common benchmark definition.

We should not compute ratio between configuration of benchmarks which are
not matching.

If we want to compute ratio between CORE AND PLATFORM then the view above should
be using CORE AND PLATFORM to filter and define the references.

*/
select otherCores.ID as ID,
 CATEGORY.category as CATEGORY,
 otherCores.NAME as NAME,
 (1.0*refCore.MAX / otherCores.MAX) as MAXRATIO,
 (1.0*refCore.MAXREGCOEF / otherCores.MAXREGCOEF) as REGRESSIONRATIO,
 PLATFORM.platform as PLATFORM,
 CORE.core as CORE,
 COMPILERKIND.compiler as COMPILER,
 COMPILER.version as COMPILERVERSION,
 TYPE.type as TYPE,
 otherCores.DATE as DATE
 from otherCores
 INNER JOIN refCore ON refCore.ID = otherCores.ID AND refCore.categoryid = otherCores.categoryid AND refCore.NAME = otherCores.NAME
 INNER JOIN CATEGORY ON CATEGORY.categoryid = otherCores.categoryid
 INNER JOIN PLATFORM ON PLATFORM.platformid = otherCores.platformid
 INNER JOIN CORE ON CORE.coreid = otherCores.coreid
 INNER JOIN COMPILER ON COMPILER.compilerid = otherCores.compilerid
 INNER JOIN COMPILERKIND ON COMPILERKIND.compilerkindid = COMPILER.compilerkindid
 INNER JOIN TYPE ON TYPE.typeid = otherCores.typeid;