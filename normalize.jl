function normalize!{T}(s::Array{T})
# Normalize s so that the squared L2 vector norm of each channel is at most 1

  (m, n, d) = size(s);
  for k=1:d;
    norm_s = vecnorm(s[:, :, k]);
    if norm_s*norm_s>1;
      s[:, :, k] /= norm_s;
    end
  end
end
