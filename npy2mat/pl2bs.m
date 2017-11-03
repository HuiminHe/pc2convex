function data = pl2bs(pts, intensity)
    data = zeros(512, 512, 660)
    for i = 1:length(pts)
        x, y, z = pts[i]
        data[x, y, z] = intensity[i]
    end
end
