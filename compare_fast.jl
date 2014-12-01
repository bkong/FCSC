require("fcsc_types.jl");
require("fcsc_learning_multi.jl");
require("fcsc_coding_fastmulti.jl");
require("fcsc_fastmulti.jl");
require("fcsc_simple.jl");
require("normalize.jl");
require("helpers.jl");
require("lagrangian.jl");

d = zeros(Float64, 10, 10, 2);
d[1:3, 1:3, 1] = [1 2 3;2 3 1;3 2 1];
d[1:3, 1:3, 2] = rotr90([1 2 3;2 3 1;3 2 1]);

X1 = [0 0 0 0 1 1 0 0 0 0;
      0 0 0 0 1 1 0 0 0 0;
      0 0 0 0 1 1 0 0 0 0;
      1 1 1 1 1 1 1 1 1 1;
      0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0;
      1 1 1 1 1 1 1 1 1 1;
      0 0 0 0 1 1 0 0 0 0;
      0 0 0 0 1 1 0 0 0 0;
      0 0 0 0 1 1 0 0 0 0];
X2 = eye(10)+rotr90(eye(10));
X1 = float64(X1); X2 = float64(X2);

opts = FCSCOpts(Float64);
opts.max_iter = 10;
opts.FIXED_D = true; opts.Dinit = d;
(nil,z_seq) = fcsc_simple(cat(3, X1, X2), 2, 5, opts);
(nil,z_bat) = fcsc_fastmulti(cat(3, X1, X2), 2, 5, opts);

@printf("\n%e\n", norm(z_seq[:]-z_bat[:]))


opts = FCSCOpts(Float64);
opts.max_iter = 10;
opts.DEBUG = true;
@time (s1,) = fcsc_simple(cat(3, X1, X2), 2, 5, opts);
@time (s2,) = fcsc_fastmulti(cat(3, X1, X2), 2, 5, opts);

@printf("\n%e\n", norm(s1[:]-s2[:]))
