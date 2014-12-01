function compute_lagrangian(data::Dict)
  return data["function"](data);
end

function compute_lagrangian_dictonly(data::Dict)
  x        = data["x"];
  d        = data["d"];
  s        = data["s"];
  lambda_s = data["lambda_s"];
  mu_s     = data["mu_s"];
  z        = data["z"];
  t        = data["t"];
  lambda_t = data["lambda_t"];
  mu_t     = data["mu_t"];
  beta     = data["beta"];

  (convH, convW, dictSize) = size(d);
  convLen = convH*convW;
  numImgs = size(x, 3);
  rerr = 0.0;
  spar = 0.0;
  cons = 0.0;
  d_fft = fft2(d);
  for j=1:numImgs;
    rerr += 0.5*vecnorm( x[:, :, j]-real(ifft2(sum(d_fft.*fft2(reshape(z[:, j], convH, convW, dictSize)), 3))) )^2;
    spar += beta*norm(t[:, j], 1);
    cons += mu_t*0.5*norm(z[:, j]-t[:, j])^2 + [lambda_t[:, j]'*(z[:, j]-t[:, j])][1];
  end
  obj = rerr + spar + cons + mu_s*0.5*norm(d[:]-s[:])^2 + [lambda_s[:]'*(d[:]-s[:])][1];

  return (obj, rerr, spar, 0.0);
end
