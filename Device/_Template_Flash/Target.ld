
PHDRS
{
   PRG PT_LOAD ;
}

PHDRS
{
   DSCR PT_LOAD ;
}

SECTIONS
{
  PrgCode :
  {
    . = ALIGN(4);

    * ( .text .text.* ) ;
    * ( EXCLUDE_FILE ( *[Ff]lash[Dd]ev.o ) .rodata .rodata.* ) ;


    . = ALIGN(4);
  } : PRG

  PrgData :
  {
    . = ALIGN(4);

    . = . + 4 ;             /* dummy for  PDT$$ptr (from  anon$$obj.o) */
    * ( .data .data.* ) ;

    . = ALIGN(4);
  }

  DevDscr :
  {
    . = ALIGN(4);

    *[Ff]lash[Dd]ev.o  ( .rodata .rodata.* ) ;

    . = ALIGN(4);
  } : DSCR
}
