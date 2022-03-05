
density_Au = 19.32;
massfraction = 0.005;
mixturedensity_Au_005 = compute_mixturedensity([density_Au,1], [massfraction,1-massfraction]);

massfraction = 0.02;
mixturedensity_Au_02 = compute_mixturedensity([density_Au,1], [massfraction,1-massfraction]);

massfraction = 0.05;
mixturedensity_Au_05 = compute_mixturedensity([density_Au,1], [massfraction,1-massfraction]);

massfraction = 0.5;
mixturedensity_Au_5 = compute_mixturedensity([density_Au,1], [massfraction,1-massfraction]);


density_Ca = 1.55;
massfraction = 0.005;
mixturedensity_Ca_005 = compute_mixturedensity([density_Ca,1], [massfraction,1-massfraction]);

massfraction = 0.02;
mixturedensity_Ca_02 = compute_mixturedensity([density_Ca,1], [massfraction,1-massfraction]);

massfraction = 0.05;
mixturedensity_Ca_05 = compute_mixturedensity([density_Ca,1], [massfraction,1-massfraction]);

massfraction = 0.5;
mixturedensity_Ca_5 = compute_mixturedensity([density_Ca,1], [massfraction,1-massfraction]);

%% Nanoparticles
density_Io = 4.93;
massfraction = 0.05;
mixturedensity_Io_05 = compute_mixturedensity([density_Io,1], [massfraction,1-massfraction]);

density_Ba = 4.5;
massfraction = 0.05;
mixturedensity_Ba_05 = compute_mixturedensity([density_Ba,1], [massfraction,1-massfraction]);

density_Gd = 7.90;
massfraction = 0.05;
mixturedensity_Gd_05 = compute_mixturedensity([density_Gd,1], [massfraction,1-massfraction]);

density_Yb = 6.973;
massfraction = 0.05;
mixturedensity_Yb_05 = compute_mixturedensity([density_Yb,1], [massfraction,1-massfraction]);

density_Ta = 16.65;
massfraction = 0.05;
mixturedensity_Ta_05 = compute_mixturedensity([density_Ta,1], [massfraction,1-massfraction]);

density_Au = 19.32;
massfraction = 0.05;
mixturedensity_Au_05 = compute_mixturedensity([density_Au,1], [massfraction,1-massfraction]);

density_Bi = 9.78;
massfraction = 0.05;
mixturedensity_Bi_05 = compute_mixturedensity([density_Bi,1], [massfraction,1-massfraction]);


%% DECT
density_Ca = 1.55;
mass_Ca = 0.05; % 50mg/ml
volume_Ca = mass_Ca/density_Ca;
massfraction_Ca_50mgpml = mass_Ca/(mass_Ca + 1 - volume_Ca);
mixturedensity_Ca_50mgpml = compute_mixturedensity([density_Ca,1], [massfraction_Ca_50mgpml,1-massfraction_Ca_50mgpml]);

density_Ca = 1.55;
mass_Ca = 0.15; % 150mg/ml
volume_Ca = mass_Ca/density_Ca;
massfraction_Ca_150mgpml = mass_Ca/(mass_Ca + 1 - volume_Ca);
mixturedensity_Ca_150mgpml = compute_mixturedensity([density_Ca,1], [massfraction_Ca_150mgpml,1-massfraction_Ca_150mgpml]);

density_Ca = 1.55;
mass_Ca = 0.3; % 300mg/ml
volume_Ca = mass_Ca/density_Ca;
massfraction_Ca_300mgpml = mass_Ca/(mass_Ca + 1 - volume_Ca);
mixturedensity_Ca_300mgpml = compute_mixturedensity([density_Ca,1], [massfraction_Ca_300mgpml,1-massfraction_Ca_300mgpml]);

density_Io = 4.93;
mass_Io = 0.005; % 5mg/ml
volume_Io = mass_Io/density_Io;
massfraction_Io_5mgpml = mass_Io/(mass_Io + 1 - volume_Io);
mixturedensity_Io_5mgpml = compute_mixturedensity([density_Io,1], [massfraction_Io_5mgpml,1-massfraction_Io_5mgpml]);

density_Io = 4.93;
mass_Io = 0.01; % 10mg/ml
volume_Io = mass_Io/density_Io;
massfraction_Io_10mgpml = mass_Io/(mass_Io + 1 - volume_Io);
mixturedensity_Io_10mgpml = compute_mixturedensity([density_Io,1], [massfraction_Io_10mgpml,1-massfraction_Io_10mgpml]);

density_Io = 4.93;
mass_Io = 0.02; % 20mg/ml
volume_Io = mass_Io/density_Io;
massfraction_Io_20mgpml = mass_Io/(mass_Io + 1 - volume_Io);
mixturedensity_Io_20mgpml = compute_mixturedensity([density_Io,1], [massfraction_Io_20mgpml,1-massfraction_Io_20mgpml]);

density_Io = 4.93;
mass_Io = 0.05; % 20mg/ml
volume_Io = mass_Io/density_Io;
massfraction_Io_50mgpml = mass_Io/(mass_Io + 1 - volume_Io);
mixturedensity_Io_50mgpml = compute_mixturedensity([density_Io,1], [massfraction_Io_50mgpml,1-massfraction_Io_50mgpml]);
