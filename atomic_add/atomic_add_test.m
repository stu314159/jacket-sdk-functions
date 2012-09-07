A = gones(4,4) % A before

ind_vec = [1 3 11 3 14];
update_vec = [1.0 2.0 3.0 4.0 5.0];

ind_vec = gsingle(ind_vec);
update_vec = gsingle(update_vec);

A(ind_vec) = A(ind_vec)+update_vec % A after

A = gones(4,4) % Re-set A

A = vecUpdateAtomicAdd(A,ind_vec,update_vec)



