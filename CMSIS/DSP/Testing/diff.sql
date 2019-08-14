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

*/
select temp.otherCores.ID as ID,
 CATEGORY.category as CATEGORY,
 temp.otherCores.NAME as NAME,
 PLATFORM.platform as PLATFORM,
 CORE.core as CORE,
 COMPILERKIND.compiler as COMPILER,
 COMPILER.version as COMPILERVERSION,
 TYPE.type as TYPE,
 temp.otherCores.DATE as DATE,
 (1.0*temp.refCore.MAX / temp.otherCores.MAX) as MAXRATIO,
 (1.0*temp.refCore.MAXREGCOEF / temp.otherCores.MAXREGCOEF) as REGRESSIONRATIO
 from temp.otherCores
 INNER JOIN temp.refCore USING(ID,categoryid,NAME)
 INNER JOIN CATEGORY USING(categoryid)
 INNER JOIN PLATFORM USING(platformid)
 INNER JOIN CORE USING(coreid)
 INNER JOIN COMPILER USING(compilerid)
 INNER JOIN COMPILERKIND USING(compilerkindid)
 INNER JOIN TYPE USING(typeid)