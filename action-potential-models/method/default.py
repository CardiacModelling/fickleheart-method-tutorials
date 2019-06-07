###############################################################################
## Defining Model Conversion, conductance, ion conc etc.
###############################################################################

# 'ina.s', 'ical.s' 'ikr.s', 'iks.s', 'ito.s', 'inaca.s', 'ik1.s'
# 'inak.s', 'if.s'

# modle_current = {'mmt_file_name': current_dictionary
# current_dictionary = { conductance_scaling: current_in_mmt }
# different model might call current differently...
# this was mainly for voltage clamp simulation...

hund_2004_current = {
    'ina.s': 'ina.ina',
    'ical.s': 'ical.ical',
    'ikr.s': 'ikr.ikr',
    'iks.s': 'iks.iks',
    'ito.s': 'ito.ito',
    'inaca.s': 'inaca.inaca',
    'ik1.s': 'ik1.ik1',
    'inak.s': 'inak.inak',
    'if.s': '',
}

tnnp_2004_current = {
    'ina.s': 'fast_sodium_current.i_Na',
    'ical.s': 'L_type_Ca_current.i_CaL',
    'ikr.s': 'rapid_time_dependent_potassium_current.i_Kr',
    'iks.s': 'slow_time_dependent_potassium_current.i_Ks',
    'ito.s': 'transient_outward_current.i_to',
    'inaca.s': 'sodium_calcium_exchanger_current.i_NaCa',
    'ik1.s': 'inward_rectifier_potassium_current.i_K1',
    'inak.s': 'sodium_potassium_pump_current.i_NaK',
    'if.s': '',
}

fink_2008_current = {
    'ina.s': 'ina.i_Na',
    'ical.s': 'ical.i_CaL',
    'ikr.s': 'ikr.i_Kr',
    'iks.s': 'iks.i_Ks',
    'ito.s': 'ito.i_to',
    'inaca.s': 'inaca.i_NaCa',
    'ik1.s': 'ik1.i_K1',
    'inak.s': 'inak.i_NaK',
    'if.s': '',
}

ohara_2011_current = {
    'ina.s': 'ina.INa',
    'ical.s': 'ical.ICaL',
    'ikr.s': 'ikr.IKr',
    'iks.s': 'iks.IKs',
    'ito.s': 'ito.Ito',
    'inaca.s': 'inaca.INaCa',
    'ik1.s': 'ik1.IK1',
    'inak.s': 'inak.INaK',
    'if.s': '',
}

model_current = {
    'hund-2004.mmt': hund_2004_current,
    'tnnp-2004.mmt': tnnp_2004_current,
    'tnnp-2004-epi.mmt': tnnp_2004_current,
    'fink-2008.mmt': fink_2008_current,
    'ohara-2011.mmt': ohara_2011_current,
}


# modle_conductance = {'mmt_file_name': condutance_dictionary
# conductance_dictionary = { conductance_scaling: condutance_in_mmt }
# different model might call conductance differently...
# this is mainly for current clamp simulation (AP simulation)

hund_2004_conductance = {
    'ina.s': 'ina.g_Na',
    'ical.s': 'ical.pca',
    'ikr.s': 'ikr.gkr_const',
    'iks.s': 'iks.gks',
    'ito.s': 'ito.gitodv',
    'inaca.s': 'inaca.NCXmax',
    'ik1.s': 'ik1.gk1',
    'inak.s': 'inak.ibarnak',
    'if.s': '',
}

tnnp_2004_conductance = {
    'ina.s': 'fast_sodium_current.g_Na',
    'ical.s': 'L_type_Ca_current.g_CaL',
    'ikr.s': 'rapid_time_dependent_potassium_current.g_Kr',
    'iks.s': 'slow_time_dependent_potassium_current.g_Ks',
    'ito.s': 'transient_outward_current.g_to',
    'inaca.s': 'sodium_calcium_exchanger_current.K_NaCa',
    'ik1.s': 'inward_rectifier_potassium_current.g_K1',
    'inak.s': 'sodium_potassium_pump_current.P_NaK',
    'if.s': '',
}

fink_2008_conductance = {
    'ina.s': 'ina.g_Na',
    'ical.s': 'ical.g_CaL',
    'ikr.s': 'ikr.g_Kr_0',
    'iks.s': 'iks.g_Ks',
    'ito.s': 'ito.g_to',
    'inaca.s': 'inaca.K_NaCa',
    'ik1.s': 'ik1.g_K1_0',
    'inak.s': 'inak.P_NaK',
    'if.s': '',
}

ohara_2011_conductance = {
    'ina.s': 'ina.GNa',
    'ical.s': 'ical.PCa',
    'ikr.s': 'ikr.GKr',
    'iks.s': 'iks.GKs',
    'ito.s': 'ito.Gto',
    'inaca.s': 'inaca.Gncx',
    'ik1.s': 'ik1.GK1',
    'inak.s': 'inak.Pnak',
    'if.s': '',
}

model_conductance = {
    'hund-2004.mmt': hund_2004_conductance,
    'tnnp-2004.mmt': tnnp_2004_conductance,
    'tnnp-2004-epi.mmt': tnnp_2004_conductance,
    'fink-2008.mmt': fink_2008_conductance,
    'ohara-2011.mmt': ohara_2011_conductance,
}


# mainly for voltage clamp -- to clamp the ion concentration too.
model_ion = {
    'hund-2004.mmt': [('sodium.Na_i',10),
                    ('ion.Na_o',150),
                    ('potassium.K_i',110),
                    ('ion.K_o',4),
                    ('calcium.Ca_i',1e-5),
                    ('ion.Ca_o',1.2),
                    ('chloride.Cl_i',15),
                    ('ion.Cl_o',100),
                    ], # Check cloride too...
    'tnnp-2004.mmt': [('sodium_dynamics.Na_i',10),
                    ('sodium_dynamics.Na_o',150),
                    ('potassium_dynamics.K_i',110),
                    ('potassium_dynamics.K_o',4),
                    ('calcium_dynamics.Ca_i',1e-5),
                    ('calcium_dynamics.Ca_o',1.2),
                    ],
    'tnnp-2004-epi.mmt': [('sodium_dynamics.Na_i',10),
                    ('sodium_dynamics.Na_o',150),
                    ('potassium_dynamics.K_i',110),
                    ('potassium_dynamics.K_o',4),
                    ('calcium_dynamics.Ca_i',1e-5),
                    ('calcium_dynamics.Ca_o',1.2),
                    ],
    'fink-2008.mmt': [('sodium.Na_i',10),
                    ('ion.Na_o',150),
                    ('potassium.K_i',110),
                    ('ion.K_o',4),
                    ('calcium.Ca_i',1e-5),
                    ('ion.Ca_o',1.2),
                    ],
    'ohara-2011.mmt': [('sodium.Nai',10),
                    ('extra.Nao',150),
                    ('potassium.Ki',110),
                    ('extra.Ko',4),
                    ('calcium.Cai',1e-5),
                    ('extra.Cao',1.2),
                    ],
    }

model_stim_amp = {
    # 'mmt_file_name': (stim_amp, value)
    'tnnp-2004.mmt': ('membrane.stim_amplitude', -52),
    'tnnp-2004-epi.mmt': ('membrane.stim_amplitude', -52),
    'fink-2008.mmt': ('stimulus.stim_amplitude', -12),
    'ohara-2011.mmt': ('stimulus.amplitude', -80),
    }

model_stim_setup = {
    # 'mmt_file_name': (stim_dur, stim_off, cycle_length, norm_stim_amp)
    'tnnp-2004.mmt': (1, 50, 1000, 1),
    'tnnp-2004-epi.mmt': (1, 50, 1000, 1),
    'fink-2008.mmt': (1, 50, 1000, 5),
    'ohara-2011.mmt': (1, 50, 1000, 1),
    }
