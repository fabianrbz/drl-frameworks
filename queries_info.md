| Query ID | Status | Index Advisor (Dexter) | Comments |
| --- | --- | --- | --- |
|  1  | --- | public.lineitem (l_partkey) | --- |
|  2  | --- |  public.lineitem (l_comment) | --- |
|  3  | --- |  public.lineitem (l_shipdate) | --- |
|  4  | --- | public.lineitem (l_discount) | using `<` makes psql use parallel scan |
|  5  | --- | public.lineitem (l_partkey) | idem |
|  6  | --- | public.lineitem (l_suppkey) | anything else was useless, has to be someting under 100 |
|  7  | --- |  public.lineitem (l_receiptdate) | --- |
|  8  | --- | public.lineitem (l_extendedprice) | --- |
|  9  | --- |  public.lineitem (l_tax) | else fails |
| 10  | --- |  public.lineitem (l_orderkey) | --- |
