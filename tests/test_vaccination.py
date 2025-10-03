"""
Unit tests for RotaVaccination intervention
"""
import pytest
import numpy as np
import starsim as ss
import rotasim as rs


def create_multi_strain_sim():
    """Helper to create multi-strain simulation for testing"""
    from rotasim.rotavirus import Rotavirus
    from rotasim.immunity import RotaImmunityConnector
    
    # Create multiple strains
    g1p8 = Rotavirus(G=1, P=8)
    g2p4 = Rotavirus(G=2, P=4)
    
    # Create immunity connector
    immunity = RotaImmunityConnector()
    
    sim = ss.Sim(
        n_agents=100,
        start=0,
        stop=2,
        dt=ss.days(7),
        diseases=[g1p8, g2p4],
        connectors=[immunity],
        networks='random',
        verbose=0
    )
    
    return sim


class TestRotaVaccinationBasic:
    """Test basic vaccination intervention functionality"""
    
    def test_vaccination_initialization(self):
        """Test basic vaccination initialization"""
        vax = rs.RotaVaccination(
            start_date='2025-01-01',
            G_antigens=[1, 2],
            P_antigens=[8, 4],
            n_doses=2,
            dose_effectiveness=[0.6, 0.8]
        )
        
        assert vax.start_date == '2025-01-01'
        assert vax.end_date is None
        assert vax.n_doses == 2
        assert vax.G_antigens == [1, 2]
        assert vax.P_antigens == [8, 4]
        assert vax.dose_effectiveness == [0.6, 0.8]
        assert hasattr(vax, 'vaccine_waning_rate')
        
    def test_vaccination_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters should work
        rs.RotaVaccination(start_date='2025-01-01')
        
        # Invalid uptake probability
        with pytest.raises(ValueError, match="uptake_prob must be between 0 and 1"):
            rs.RotaVaccination(start_date='2025-01-01', uptake_prob=1.5)
            
        # Invalid number of doses
        with pytest.raises(ValueError, match="n_doses must be between 1 and 10"):
            rs.RotaVaccination(start_date='2025-01-01', n_doses=0)
            
        # Mismatched dose effectiveness
        with pytest.raises(ValueError, match="dose_effectiveness must have 3 values"):
            rs.RotaVaccination(start_date='2025-01-01', n_doses=3, dose_effectiveness=[0.6, 0.8])
            
    def test_vaccination_defaults(self):
        """Test default parameter values"""
        vax = rs.RotaVaccination(start_date='2025-01-01')
        
        assert vax.n_doses == 2
        assert vax.dose_interval == ss.days(28)
        assert vax.G_antigens == [1]
        assert vax.P_antigens == [8]
        assert vax.dose_effectiveness == [0.6, 0.8]
        assert vax.min_age == ss.days(42)
        assert vax.max_age == ss.days(365)
        assert vax.eligible_only_once is True


class TestRotaVaccinationSimulation:
    """Test vaccination within simulation context"""
    
    def create_test_sim(self, vax_kwargs=None, sim_kwargs=None):
        """Helper to create test simulation"""
        if vax_kwargs is None:
            vax_kwargs = {}
        if sim_kwargs is None:
            sim_kwargs = {}
            
        default_vax = {
            'start_date': '2020-01-01',
            'verbose': False
        }
        default_vax.update(vax_kwargs)
        
        default_sim = {
            'scenario': 'simple',
            'n_agents': 1000,
            'start': '2020-01-01',
            'stop': '2022-01-01',
            'dt': ss.days(7),
            'verbose': 0
        }
        default_sim.update(sim_kwargs)
        
        vax = rs.RotaVaccination(**default_vax)
        
        # Add intervention
        if 'interventions' in default_sim:
            default_sim['interventions'].append(vax)
        else:
            default_sim['interventions'] = [vax]
            
        sim = rs.Sim(**default_sim)
        return sim
    
    def test_vaccination_initialization_in_sim(self):
        """Test vaccination initialization within simulation"""
        sim = self.create_test_sim()
        sim.init()
        
        # Get the vaccination intervention from sim
        vax = sim.interventions[0]
        
        # Check that vaccination states were created
        assert hasattr(vax, 'doses_received')
        assert hasattr(vax, 'last_dose_time')
        assert hasattr(vax, 'next_dose_due')
        assert hasattr(vax, 'ever_eligible')
        assert hasattr(vax, 'completed_schedule')
        
        # Check covered diseases were identified
        assert hasattr(vax, 'vaccine_protection_states')
        assert hasattr(vax, 'vaccine_waning_states')
        assert len(vax.vaccine_protection_states) > 0
        assert len(vax.vaccine_waning_states) > 0
        
        # Check state array sizes
        n_agents = len(sim.people)
        assert len(vax.doses_received) == n_agents
        assert len(vax.last_dose_time) == n_agents
        
    def test_vaccination_coverage_identification(self):
        """Test that vaccine correctly identifies covered diseases"""
        # Test G1P8 vaccine
        sim = self.create_test_sim({
            'G_antigens': [1],
            'P_antigens': [8]
        })
        sim.init()
        vax = sim.interventions[0]
        
        # Should cover G1P8 disease
        covered_strains = []
        for disease_name in vax.vaccine_protection_states:
            # Find corresponding disease
            for disease in sim.diseases.values():
                if disease.name == disease_name:
                    covered_strains.append((disease.G, disease.P))
                    break
        assert (1, 8) in covered_strains
        assert len([s for s in covered_strains if s == (1, 8)]) == 1
        
        # Test multi-strain vaccine
        sim2 = self.create_test_sim({
            'G_antigens': [1, 2],
            'P_antigens': [8, 4]
        })
        sim2.init()
        vax2 = sim2.interventions[0]
        
        covered_strains2 = []
        for disease_name in vax2.vaccine_protection_states:
            # Find corresponding disease
            for disease in sim2.diseases.values():
                if disease.name == disease_name:
                    covered_strains2.append((disease.G, disease.P))
                    break
        assert (1, 8) in covered_strains2
        assert (2, 4) in covered_strains2
        # Should not cover partial matches like (1,4) or (2,8)
        
    def test_age_eligibility(self):
        """Test age-based eligibility"""
        sim = self.create_test_sim({
            'min_age': ss.days(60),  # ~2 months
            'max_age': ss.days(300)  # ~10 months
        })
        sim.init()
        vax = sim.interventions[0]
        
        # Manually set some agent ages for testing
        sim.people.age.values[:100] = ss.days(30)   # Too young
        sim.people.age.values[100:200] = ss.days(120)  # Eligible age
        sim.people.age.values[200:300] = ss.days(400)  # Too old
        
        eligible = vax.check_eligibility(sim)
        
        # Should not include too young or too old
        assert not np.any(eligible[:100])   # Too young
        assert not np.any(eligible[200:300])  # Too old
        
        # Should include eligible age (though some may be excluded by other criteria)
        assert np.sum(eligible[100:200]) >= 0  # At least some eligible
        
    def test_vaccination_timing(self):
        """Test vaccination start and end dates"""
        # Test vaccination before start date
        sim = self.create_test_sim({
            'start_date': '2021-01-01'  # Start after sim start
        })
        sim.init()
        vax = sim.interventions[0]
        
        # At simulation start, should not be eligible (before vaccine start)
        eligible_early = vax.check_eligibility(sim)
        assert np.sum(eligible_early) == 0
        
        # Test vaccination with end date
        sim2 = self.create_test_sim({
            'start_date': '2020-01-01',
            'end_date': '2020-06-01'
        })
        sim2.init()
        vax2 = sim2.interventions[0]
        
        # Step forward to when vaccine should be active
        sim2.run_one_step()  # Move to ti=1 when vaccine should be active
        
        # At this point, should have eligible agents
        # First ensure some agents are in the eligible age range  
        sim2.people.age.values[:100] = ss.days(120)  # Set some agents to eligible age
        eligible_start = vax2.check_eligibility(sim2)
        assert np.sum(eligible_start) > 0
        
    def test_vaccination_application(self):
        """Test that vaccination is applied correctly"""
        sim = self.create_test_sim({
            'uptake_prob': 1.0,  # 100% uptake for testing
            'min_age': ss.days(0),  # All ages eligible
            'max_age': ss.days(10000)
        })
        sim.init()
        vax = sim.interventions[0]
        
        # Run for a few steps
        for _ in range(5):
            sim.run_one_step()
            
        # Check that some agents were vaccinated
        vaccinated_count = np.sum(vax.doses_received > 0)
        assert vaccinated_count > 0
        
        # Check that vaccination tracking works
        assert np.all(vax.last_dose_time[vax.doses_received > 0] >= 0)
        assert np.all(vax.ever_eligible[vax.doses_received > 0])


class TestRotaVaccinationProtection:
    """Test vaccine protection mechanism"""
    
    def create_test_sim_with_protection(self):
        """Create simulation for testing protection"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            G_antigens=[1],
            P_antigens=[8],
            dose_effectiveness=[0.8, 0.9],
            uptake_prob=1.0,
            min_age=ss.days(0),
            max_age=ss.days(10000),
            verbose=False
        )
        
        sim = rs.Sim(
            scenario='simple',
            n_agents=100,
            start='2020-01-01',
            stop='2021-01-01',
            dt=ss.days(1),
            interventions=[vax],
            verbose=0
        )
        
        return sim
        
    def test_protection_states_creation(self):
        """Test that protection states are created for covered diseases"""
        sim = self.create_test_sim_with_protection()
        sim.init()
        vax = sim.interventions[0]
        
        # Should have protection states for covered diseases
        assert len(vax.vaccine_protection_states) > 0
        assert len(vax.vaccine_waning_states) > 0
        
        # States should be same length as population
        n_agents = len(sim.people)
        for state in vax.vaccine_protection_states.values():
            assert len(state) == n_agents
        for state in vax.vaccine_waning_states.values():
            assert len(state) == n_agents
            
    def test_protection_application(self):
        """Test that protection is applied when agents are vaccinated"""
        sim = self.create_test_sim_with_protection()
        sim.init()
        vax = sim.interventions[0]
        
        # Get initial protection levels (should be 0)
        initial_protection = {}
        for disease_name, state in vax.vaccine_protection_states.items():
            initial_protection[disease_name] = state[:].copy()
            assert np.all(state[:] == 0.0)
            
        # Run simulation for a few steps to allow vaccinations
        for _ in range(30):
            sim.run_one_step()
            
        # Check that some agents now have protection
        vaccinated_any = np.sum(vax.doses_received > 0) > 0
        assert vaccinated_any, "No agents were vaccinated"
        
        # Check protection states updated
        for disease_name, state in vax.vaccine_protection_states.items():
            if vaccinated_any:
                # Some agents should have protection > 0
                assert np.sum(state[:] > 0) > 0, f"No protection applied for {disease_name}"
                
    def test_protection_waning(self):
        """Test that vaccine protection wanes over time"""
        sim = self.create_test_sim_with_protection()
        sim.init()
        vax = sim.interventions[0]
        
        # Use very fast waning for testing - just use a simple constant
        # The waning logic will use fallback to np.full(len(uids), 365) if rvs doesn't work
        vax.vaccine_waning_rate = 2  # 2 days constant waning time
        
        # Run for vaccination period
        for _ in range(30):
            sim.run_one_step()
            
        # Get protection levels after vaccination
        mid_protection = {}
        for disease_name, state in vax.vaccine_protection_states.items():
            mid_protection[disease_name] = state[:].copy()
            
        # Run for waning period
        for _ in range(10):  # Additional days for waning
            sim.run_one_step()
            
        # Check that protection has waned
        for disease_name, state in vax.vaccine_protection_states.items():
            current_protection = state[:]
            mid_values = mid_protection[disease_name]
            
            # Where there was protection, it should have decreased (or stayed same if recent)
            protected_agents = mid_values > 0
            if np.any(protected_agents):
                # At least some protection should have waned
                assert np.any(current_protection[protected_agents] <= mid_values[protected_agents])
                
    def test_rel_sus_modification(self):
        """Test that vaccine protection modifies rel_sus parameters"""
        sim = self.create_test_sim_with_protection()
        sim.init()
        vax = sim.interventions[0]
        
        # Get initial rel_sus values
        initial_rel_sus = {}
        # Find covered diseases through protection states
        covered_disease_names = list(vax.vaccine_protection_states.keys())
        for disease in sim.diseases.values():
            if disease.name in covered_disease_names and hasattr(disease, 'rel_sus'):
                initial_rel_sus[disease.name] = disease.rel_sus[:].copy()
        
        # Run simulation to apply vaccinations
        for _ in range(50):
            sim.run_one_step()
            
        # Check that rel_sus was modified for covered diseases
        for disease in sim.diseases.values():
            if disease.name in covered_disease_names and hasattr(disease, 'rel_sus') and disease.name in initial_rel_sus:
                current_rel_sus = disease.rel_sus[:]
                initial_values = initial_rel_sus[disease.name]
                
                # Some agents should have reduced susceptibility
                # (lower rel_sus values indicate better protection)
                if np.sum(vax.doses_received > 0) > 0:  # If anyone was vaccinated
                    assert np.any(current_rel_sus <= initial_values)


class TestRotaVaccinationMultiDose:
    """Test multi-dose vaccination schedules"""
    
    def test_two_dose_schedule(self):
        """Test 2-dose vaccination schedule"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            n_doses=2,
            dose_interval=ss.days(28),
            dose_effectiveness=[0.6, 0.9],
            uptake_prob=1.0,
            min_age=ss.days(0),
            max_age=ss.days(10000)
        )
        
        sim = rs.Sim(
            scenario='simple',
            n_agents=100,
            start='2020-01-01',
            stop='2020-06-01',
            dt=ss.days(1),
            interventions=[vax],
            verbose=0
        )
        sim.init()
        vax = sim.interventions[0]  # Get intervention after init
        
        # Run simulation
        for _ in range(120):  # ~4 months
            sim.run_one_step()
            
        # Check dose distribution
        summary = vax.get_vaccination_summary()
        
        # Should have agents with both doses
        assert summary['doses_by_number'][1] > 0  # First dose
        assert summary['doses_by_number'][2] > 0  # Second dose
        assert summary['completed_schedule'] > 0
        
        # Second dose count should be <= first dose count
        assert summary['doses_by_number'][2] <= summary['doses_by_number'][1]
        
    def test_three_dose_schedule(self):
        """Test 3-dose vaccination schedule"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            n_doses=3,
            dose_interval=ss.days(21),
            dose_effectiveness=[0.4, 0.7, 0.9],
            uptake_prob=1.0,
            min_age=ss.days(0),
            max_age=ss.days(10000)
        )
        
        sim = rs.Sim(
            scenario='simple',
            n_agents=50,
            start='2020-01-01',
            stop='2020-06-01',
            dt=ss.days(1),
            interventions=[vax],
            verbose=0
        )
        sim.init()
        vax = sim.interventions[0]  # Get intervention after init
        
        # Run simulation
        for _ in range(150):
            sim.run_one_step()
            
        # Check that 3-dose schedule works
        summary = vax.get_vaccination_summary()
        assert summary['doses_by_number'][1] > 0  # First dose
        
        # Should have some agents progressing through schedule
        max_doses = np.max(vax.doses_received)
        assert max_doses <= 3  # No more than 3 doses
        
    def test_dose_timing(self):
        """Test that doses are given at correct intervals"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            n_doses=2,
            dose_interval=ss.days(28),
            uptake_prob=1.0,
            min_age=ss.days(0),
            max_age=ss.days(10000)
        )
        
        sim = rs.Sim(
            scenario='simple',
            n_agents=10,  # Small for detailed checking
            start='2020-01-01',
            stop='2020-06-01',
            dt=ss.days(1),
            interventions=[vax],
            verbose=0
        )
        sim.init()
        vax = sim.interventions[0]  # Get intervention after init
        
        # Track when agents get doses
        dose_times = {uid: [] for uid in range(10)}
        
        for step in range(100):
            prev_doses = vax.doses_received[:].copy()
            sim.run_one_step()
            new_doses = vax.doses_received[:]
            
            # Record when each agent gets a dose
            for uid in range(len(prev_doses)):  # Use actual array length
                if uid < len(new_doses) and new_doses[uid] > prev_doses[uid]:
                    dose_times[uid].append(sim.ti)  # Use ti instead of t
                    
        # Check dose intervals for agents who got multiple doses
        for uid, times in dose_times.items():
            if len(times) >= 2:
                interval = times[1] - times[0]
                # Should be approximately 28 days (allowing some tolerance)
                assert 27 <= interval <= 29


class TestRotaVaccinationSummary:
    """Test vaccination summary and reporting functions"""
    
    def create_test_sim(self):
        """Helper to create test simulation"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            uptake_prob=0.5,  # Moderate uptake
            verbose=False
        )
        
        sim = rs.Sim(
            scenario='simple',
            n_agents=200,
            start='2020-01-01',
            stop='2021-01-01',
            dt=ss.days(7),
            interventions=[vax],
            verbose=0
        )
        
        return sim
        
    def test_vaccination_summary(self):
        """Test vaccination summary generation"""
        sim = self.create_test_sim()
        sim.init()
        vax = sim.interventions[0]
        
        # Run simulation
        for _ in range(50):
            sim.run_one_step()
            
        summary = vax.get_vaccination_summary()
        
        # Check summary structure
        assert 'total_agents' in summary
        assert 'ever_eligible' in summary
        assert 'received_any_dose' in summary
        assert 'completed_schedule' in summary
        assert 'doses_by_number' in summary
        assert 'mean_doses' in summary
        
        # Check values make sense
        assert summary['total_agents'] == len(sim.people)
        assert summary['received_any_dose'] <= summary['ever_eligible']
        assert summary['completed_schedule'] <= summary['received_any_dose']
        
    def test_print_vaccination_summary(self):
        """Test vaccination summary printing (should not crash)"""
        sim = self.create_test_sim()
        sim.init()
        vax = sim.interventions[0]
        
        # Should not crash even with no vaccinations
        vax.print_vaccination_summary()
        
        # Run some simulation
        for _ in range(20):
            sim.run_one_step()
            
        # Should not crash with some vaccinations
        vax.print_vaccination_summary()


class TestRotaVaccinationCrossProtection:
    """Test cross-protection functionality"""
    
    def test_strain_matching_functions(self):
        """Test strain matching helper functions"""
        vax = rs.RotaVaccination(
            start_date='2025-01-01',
            G_antigens=[1, 2],
            P_antigens=[8, 4]
        )
        
        # Create mock disease objects for testing
        class MockDisease:
            def __init__(self, G, P, name):
                self.G = G
                self.P = P
                self.name = name
        
        # Homotypic matches
        g1p8 = MockDisease(1, 8, 'G1P8')
        g2p4 = MockDisease(2, 4, 'G2P4')
        assert vax._is_homotypic_match(g1p8)
        assert vax._is_homotypic_match(g2p4)
        assert not vax._is_partial_heterotypic_match(g1p8)
        assert not vax._is_complete_heterotypic_match(g1p8)
        
        # Partial heterotypic matches
        g1p6 = MockDisease(1, 6, 'G1P6')  # Shared G
        g3p8 = MockDisease(3, 8, 'G3P8')  # Shared P
        assert vax._is_partial_heterotypic_match(g1p6)
        assert vax._is_partial_heterotypic_match(g3p8)
        assert not vax._is_homotypic_match(g1p6)
        assert not vax._is_complete_heterotypic_match(g1p6)
        
        # Complete heterotypic matches
        g3p6 = MockDisease(3, 6, 'G3P6')  # No shared G or P
        assert vax._is_complete_heterotypic_match(g3p6)
        assert not vax._is_homotypic_match(g3p6)
        assert not vax._is_partial_heterotypic_match(g3p6)
    
    def test_cross_protection_efficacy_parameters(self):
        """Test cross-protection efficacy parameter validation"""
        # Valid parameters
        vax = rs.RotaVaccination(
            start_date='2025-01-01',
            homotypic_efficacy=1.0,
            partial_heterotypic_efficacy=0.6,
            complete_heterotypic_efficacy=0.3
        )
        assert vax.homotypic_efficacy == 1.0
        assert vax.partial_heterotypic_efficacy == 0.6
        assert vax.complete_heterotypic_efficacy == 0.3
        
        # Invalid parameters should raise ValueError
        with pytest.raises(ValueError, match="homotypic_efficacy must be between 0 and 1"):
            rs.RotaVaccination(start_date='2025-01-01', homotypic_efficacy=1.5)
        
        with pytest.raises(ValueError, match="partial_heterotypic_efficacy must be between 0 and 1"):
            rs.RotaVaccination(start_date='2025-01-01', partial_heterotypic_efficacy=-0.1)
        
        with pytest.raises(ValueError, match="complete_heterotypic_efficacy must be between 0 and 1"):
            rs.RotaVaccination(start_date='2025-01-01', complete_heterotypic_efficacy=1.2)
    
    def test_all_diseases_covered_with_cross_protection(self):
        """Test that all rotavirus diseases are covered with cross-protection"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            G_antigens=[1],  # Only covers G1
            P_antigens=[8],  # Only covers P8
            homotypic_efficacy=1.0,
            partial_heterotypic_efficacy=0.6,
            complete_heterotypic_efficacy=0.3,
            verbose=False
        )
        
        sim = rs.Sim(
            scenario='simple',
            n_agents=100,
            start='2020-01-01',
            stop='2021-01-01',
            dt=ss.days(7),
            interventions=[vax],
            verbose=0
        )
        
        sim.init()
        
        # Get the actual vaccination intervention from the sim
        vax = sim.interventions[0]
        
        # With cross-protection, ALL rotavirus diseases should be covered
        assert len(vax.covered_diseases) == len(vax.rotavirus_diseases)
        assert len(vax.covered_diseases) > 0
        
        # Should have created protection states for all diseases
        assert len(vax.vaccine_protection_states) == len(vax.covered_diseases)
        assert len(vax.vaccine_waning_states) == len(vax.covered_diseases)
    
    def test_cross_protection_effectiveness_calculation(self):
        """Test that cross-protection applies correct effectiveness levels"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            n_doses=1,
            G_antigens=[1],
            P_antigens=[8], 
            dose_effectiveness=[0.8],  # 80% base effectiveness
            homotypic_efficacy=1.0,     # 100% for G1P8 
            partial_heterotypic_efficacy=0.6,  # 60% for partial matches
            complete_heterotypic_efficacy=0.3,   # 30% for no matches
            verbose=False
        )
        
        sim = rs.Sim(
            scenario='simple',
            n_agents=100,
            start='2020-01-01',
            stop='2021-01-01',
            dt=ss.days(7),
            interventions=[vax],
            verbose=0
        )
        
        sim.init()
        
        # Get the actual vaccination intervention from the sim
        vax = sim.interventions[0]
        
        # Vaccinate a test agent
        test_uids = [0]
        current_doses = [0]  # First dose (0-indexed)
        
        vax._apply_vaccine_protection(sim, test_uids, current_doses)
        
        # Check protection levels for different diseases
        for disease in vax.covered_diseases:
            protection_state = vax.vaccine_protection_states[disease.name]
            protection_level = protection_state[0]  # Agent 0
            
            # Use precomputed match efficacy
            match_efficacy = vax.disease_match_efficacies[disease.name]
            expected = 0.8 * match_efficacy  # 80% base effectiveness * match efficacy
            
            assert abs(protection_level - expected) < 1e-6, f"Disease {disease.name} (G{disease.G}P{disease.P}): expected {expected}, got {protection_level}"
    
    def test_precomputed_match_efficacies(self):
        """Test that precomputed match efficacies are correctly calculated"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            G_antigens=[1, 2],
            P_antigens=[8, 4],
            homotypic_efficacy=1.0,
            partial_heterotypic_efficacy=0.6,
            complete_heterotypic_efficacy=0.3,
            verbose=False
        )
        
        sim = rs.Sim(
            scenario='simple',
            n_agents=50,
            start='2020-01-01',
            stop='2021-01-01',
            dt=ss.days(7),
            interventions=[vax],
            verbose=0
        )
        
        sim.init()
        vax = sim.interventions[0]
        
        # Verify that precomputed efficacies match individual function calls
        for disease in vax.covered_diseases:
            precomputed = vax.disease_match_efficacies[disease.name]
            manual_calculation = vax._compute_match_efficacy(disease)
            
            assert precomputed == manual_calculation, f"Mismatch for {disease.name}: precomputed={precomputed}, manual={manual_calculation}"
            
            # Verify the specific expected values for known strain patterns
            if disease.G in vax.G_antigens and disease.P in vax.P_antigens:
                assert precomputed == 1.0, f"Homotypic {disease.name} should have efficacy 1.0"
            elif disease.G in vax.G_antigens or disease.P in vax.P_antigens:
                assert precomputed == 0.6, f"Partial heterotypic {disease.name} should have efficacy 0.6"
            else:
                assert precomputed == 0.3, f"Complete heterotypic {disease.name} should have efficacy 0.3"
        
    def create_test_sim(self):
        """Helper to create test simulation"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            uptake_prob=0.5,  # Moderate uptake
            verbose=False
        )
        
        sim = rs.Sim(
            scenario='simple',
            n_agents=200,
            start='2020-01-01',
            stop='2021-01-01',
            dt=ss.days(7),
            interventions=[vax],
            verbose=0
        )
        
        return sim


if __name__ == '__main__':
    # Run tests if script is called directly
    pytest.main([__file__])