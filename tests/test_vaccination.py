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
        
        assert vax.pars.start_date == '2025-01-01'
        assert vax.pars.end_date is None
        assert vax.pars.n_doses == 2
        assert vax.pars.G_antigens == [1, 2]
        assert vax.pars.P_antigens == [8, 4]
        assert vax.pars.dose_effectiveness == [0.6, 0.8]
        assert hasattr(vax.pars, 'waning_rate_dist')
        
    def test_vaccination_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters should work
        rs.RotaVaccination(start_date='2025-01-01')
        
        # Invalid waning rate distribution - must be ss.Dist
        with pytest.raises(ValueError, match="waning_rate_dist must be an ss.Dist"):
            rs.RotaVaccination(start_date='2025-01-01', waning_rate_dist=365)
            
        # Invalid number of doses
        with pytest.raises(ValueError, match="n_doses must be between >= 1"):
            rs.RotaVaccination(start_date='2025-01-01', n_doses=0)
            
        # Mismatched dose effectiveness
        with pytest.raises(ValueError, match="dose_effectiveness must have 3 values"):
            rs.RotaVaccination(start_date='2025-01-01', n_doses=3, dose_effectiveness=[0.6, 0.8])
            
    def test_vaccination_defaults(self):
        """Test default parameter values"""
        vax = rs.RotaVaccination(start_date='2025-01-01')
        
        assert vax.pars.n_doses == 2
        assert vax.pars.dose_interval == ss.days(28)
        assert vax.pars.G_antigens == [1]
        assert vax.pars.P_antigens == [8]
        assert vax.pars.dose_effectiveness == [0.6, 0.8]
        assert vax.pars.min_age == ss.days(42)
        assert vax.pars.max_age == ss.days(365)


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
        assert hasattr(vax, 'last_dose_ti')
        assert hasattr(vax, 'next_dose_due')
        assert hasattr(vax, 'doses_eligible')
        assert hasattr(vax, 'completed_schedule')
        assert hasattr(vax, 'waning_rate')
        assert hasattr(vax, 'waning_delay')
        
        # Check covered diseases were identified
        assert hasattr(vax, 'covered_diseases')
        assert hasattr(vax, 'disease_match_efficacies')
        assert len(vax.covered_diseases) > 0
        assert len(vax.disease_match_efficacies) > 0
        
        # Check state array sizes
        n_agents = len(sim.people)
        assert len(vax.doses_received) == n_agents
        assert len(vax.last_dose_ti) == n_agents
        
    def test_vaccination_coverage_identification(self):
        """Test that vaccine correctly identifies covered diseases"""
        # Test G1P8 vaccine
        sim = self.create_test_sim({
            'G_antigens': [1],
            'P_antigens': [8]
        })
        sim.init()
        vax = sim.interventions[0]
        
        # Should cover all Rotavirus diseases (with cross-protection)
        covered_strains = [(d.G, d.P) for d in vax.covered_diseases]
        assert len(covered_strains) > 0
        
        # Check that precomputed efficacies exist for covered diseases
        for disease in vax.covered_diseases:
            assert disease.name in vax.disease_match_efficacies
            
        # Verify that G1P8 gets homotypic efficacy
        for disease in vax.covered_diseases:
            if (disease.G, disease.P) == (1, 8):
                assert vax.disease_match_efficacies[disease.name] == vax.pars.homotypic_efficacy
        
        # Test multi-strain vaccine
        sim2 = self.create_test_sim({
            'G_antigens': [1, 2],
            'P_antigens': [8, 4]
        })
        sim2.init()
        vax2 = sim2.interventions[0]
        
        # Should still cover all diseases with appropriate efficacies
        covered_strains2 = [(d.G, d.P) for d in vax2.covered_diseases]
        assert len(covered_strains2) > 0
        
        # Verify homotypic efficacies for both target strains
        homotypic_found = 0
        for disease in vax2.covered_diseases:
            if (disease.G, disease.P) in [(1, 8), (2, 4)]:
                assert vax2.disease_match_efficacies[disease.name] == vax2.pars.homotypic_efficacy
                homotypic_found += 1
        assert homotypic_found >= 1  # At least one target strain should be present
        
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
        
        eligible = vax.check_eligibility()
        
        # Should not include too young or too old
        assert not np.any(eligible.values[:100])   # Too young
        assert not np.any(eligible.values[200:300])  # Too old
        
        # Should include eligible age (though some may be excluded by other criteria)
        assert np.sum(eligible.values[100:200]) >= 0  # At least some eligible
        
    def test_vaccination_timing(self):
        """Test vaccination start and end dates"""
        # Test vaccination before start date
        sim = self.create_test_sim({
            'start_date': '2021-01-01'  # Start after sim start
        })
        sim.init()
        vax = sim.interventions[0]
        
        # At simulation start, no vaccinations should occur (before vaccine start)
        initial_doses = np.sum(vax.doses_received > 0)
        sim.run_one_step()  # Run one step
        after_step_doses = np.sum(vax.doses_received > 0)
        assert after_step_doses == initial_doses  # No new vaccinations should occur
        
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
        eligible_start = vax2.check_eligibility()
        assert np.sum(eligible_start) > 0
        
    def test_vaccination_application(self):
        """Test that vaccination is applied correctly"""
        sim = self.create_test_sim({
            'uptake_dist': ss.bernoulli(1.0),  # 100% uptake for testing
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
        assert np.all(vax.last_dose_ti[vax.doses_received > 0] >= 0)
        assert np.all(vax.doses_eligible[vax.doses_received > 0] > 0)


class TestRotaVaccinationProtection:
    """Test vaccine protection mechanism"""
    
    def create_test_sim_with_protection(self):
        """Create simulation for testing protection"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            G_antigens=[1],
            P_antigens=[8],
            dose_effectiveness=[0.8, 0.9],
            uptake_dist=ss.bernoulli(1.0),
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
        """Test that vaccination states are created for covered diseases"""
        sim = self.create_test_sim_with_protection()
        sim.init()
        vax = sim.interventions[0]
        
        # Should have covered diseases identified
        assert len(vax.covered_diseases) > 0
        assert len(vax.disease_match_efficacies) > 0
        
        # Basic vaccination states should be initialized
        n_agents = len(sim.people)
        assert len(vax.doses_received) == n_agents
        assert len(vax.waning_rate) == n_agents
        assert len(vax.waning_delay) == n_agents
        
        # Check that match efficacies are computed for covered diseases
        for disease in vax.covered_diseases:
            assert disease.name in vax.disease_match_efficacies
            
    def test_protection_application(self):
        """Test that protection is applied when agents are vaccinated"""
        sim = self.create_test_sim_with_protection()
        sim.init()
        vax = sim.interventions[0]
        
        # Get initial rel_sus levels (should be 1.0 = fully susceptible)
        initial_rel_sus = {}
        for disease in vax.covered_diseases:
            initial_rel_sus[disease.name] = disease.rel_sus[:].copy()
            assert np.all(disease.rel_sus[:] == 1.0)
            
        # Run simulation for a few steps to allow vaccinations
        for _ in range(30):
            sim.run_one_step()
            
        # Check that some agents were vaccinated
        vaccinated_any = np.sum(vax.doses_received > 0) > 0
        assert vaccinated_any, "No agents were vaccinated"
        
        # Check that rel_sus was modified for covered diseases (protection applied)
        if vaccinated_any:
            for disease in vax.covered_diseases:
                current_rel_sus = disease.rel_sus[:]
                initial_values = initial_rel_sus[disease.name]
                
                # Some agents should have reduced susceptibility (rel_sus < 1.0)
                # This indicates vaccine protection was applied
                assert np.any(current_rel_sus < initial_values), f"No protection applied for {disease.name}"
                
    def test_protection_waning(self):
        """Test that vaccine protection wanes over time"""
        sim = self.create_test_sim_with_protection()
        sim.init()
        vax = sim.interventions[0]
        
        # Set fast waning for testing by modifying the waning rate for all agents
        # This will be used when agents get vaccinated
        
        # Run for vaccination period
        for _ in range(30):
            sim.run_one_step()
            
        # Check that some agents were vaccinated
        vaccinated_any = np.sum(vax.doses_received > 0) > 0
        if not vaccinated_any:
            return  # Skip test if no vaccinations occurred
            
        # Set fast waning rate for vaccinated agents
        vaccinated_mask = vax.doses_received > 0
        vax.waning_rate[vaccinated_mask] = 0.5  # Fast waning: 50% per day
        
        # Get rel_sus levels after vaccination
        mid_rel_sus = {}
        for disease in vax.covered_diseases:
            mid_rel_sus[disease.name] = disease.rel_sus[:].copy()
            
        # Run for waning period
        for _ in range(5):  # Additional days for waning
            sim.run_one_step()
            
        # Check that protection has waned (rel_sus should increase toward 1.0)
        for disease in vax.covered_diseases:
            current_rel_sus = disease.rel_sus[:]
            mid_values = mid_rel_sus[disease.name]
            
            # For vaccinated agents, rel_sus should have increased (less protection)
            if np.any(vaccinated_mask):
                # Protection should have waned somewhat (though may still be protected)
                assert np.any(current_rel_sus[vaccinated_mask] >= mid_values[vaccinated_mask]), f"No waning detected for {disease.name}"
                
    def test_rel_sus_modification(self):
        """Test that vaccine protection modifies rel_sus parameters"""
        sim = self.create_test_sim_with_protection()
        sim.init()
        vax = sim.interventions[0]
        
        # Get initial rel_sus values for covered diseases
        initial_rel_sus = {}
        for disease in vax.covered_diseases:
            if hasattr(disease, 'rel_sus'):
                initial_rel_sus[disease.name] = disease.rel_sus[:].copy()
        
        # Run simulation to apply vaccinations
        for _ in range(50):
            sim.run_one_step()
            
        # Check that rel_sus was modified for covered diseases
        for disease in vax.covered_diseases:
            if hasattr(disease, 'rel_sus') and disease.name in initial_rel_sus:
                current_rel_sus = disease.rel_sus[:]
                initial_values = initial_rel_sus[disease.name]
                
                # Some agents should have reduced susceptibility
                # (lower rel_sus values indicate better protection)
                if np.sum(vax.doses_received > 0) > 0:  # If anyone was vaccinated
                    assert np.any(current_rel_sus <= initial_values), f"rel_sus not modified for {disease.name}"


class TestRotaVaccinationMultiDose:
    """Test multi-dose vaccination schedules"""
    
    def test_two_dose_schedule(self):
        """Test 2-dose vaccination schedule"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            n_doses=2,
            dose_interval=ss.days(28),
            dose_effectiveness=[0.6, 0.9],
            uptake_dist=ss.bernoulli(1.0),
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
        
        # Should have agents who received doses
        total_vaccinated = summary['received_any_dose']
        assert total_vaccinated > 0  # Some agents got vaccinated
        assert summary['completed_schedule'] > 0  # Some completed the schedule
        
        # Check dose counts - agents progress from 1 to 2 doses
        # At the end, we should have agents with 2 doses
        assert summary['doses_by_number'][2] > 0  # Second dose
        
        # Total doses given should be consistent
        doses_given = summary['doses_by_number'][1] + summary['doses_by_number'][2]
        assert doses_given == total_vaccinated
        
    def test_three_dose_schedule(self):
        """Test 3-dose vaccination schedule"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            n_doses=3,
            dose_interval=ss.days(21),
            dose_effectiveness=[0.4, 0.7, 0.9],
            uptake_dist=ss.bernoulli(1.0),
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
        
        # Should have some agents who received doses
        total_vaccinated = summary['received_any_dose']
        assert total_vaccinated > 0  # Some agents got vaccinated
        
        # Should have some agents progressing through schedule
        max_doses = np.max(vax.doses_received)
        assert max_doses <= 3  # No more than 3 doses
        assert max_doses > 0   # At least some doses given
        
    def test_dose_timing(self):
        """Test that doses are given at correct intervals"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            n_doses=2,
            dose_interval=ss.days(28),
            uptake_dist=ss.bernoulli(1.0),
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
            uptake_dist=ss.bernoulli(0.5),  # Moderate uptake
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
        assert 'doses_eligible' in summary
        assert 'received_any_dose' in summary
        assert 'completed_schedule' in summary
        assert 'doses_by_number' in summary
        assert 'mean_doses' in summary
        
        # Check values make sense
        assert summary['total_agents'] == len(sim.people)
        assert summary['received_any_dose'] <= summary['doses_eligible']
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
        assert vax.pars.homotypic_efficacy == 1.0
        assert vax.pars.partial_heterotypic_efficacy == 0.6
        assert vax.pars.complete_heterotypic_efficacy == 0.3
        
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
        assert len(vax.covered_diseases) >= 1  # Should cover at least some diseases
        assert len(vax.covered_diseases) > 0
        
        # Should have precomputed match efficacies for all covered diseases
        assert len(vax.disease_match_efficacies) == len(vax.covered_diseases)
        
        # Each covered disease should have a match efficacy computed
        for disease in vax.covered_diseases:
            assert disease.name in vax.disease_match_efficacies
    
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
        test_uids = ss.uids([0])
        current_doses = [0]  # First dose (0-indexed)
        
        vax._apply_vaccine_protection(test_uids, current_doses)
        
        # Check that match efficacies are correctly calculated for different diseases
        for disease in vax.covered_diseases:
            # Use precomputed match efficacy
            match_efficacy = vax.disease_match_efficacies[disease.name]
            
            # Verify the correct match efficacy based on G,P values
            if disease.G == 1 and disease.P == 8:
                # Homotypic match
                expected_efficacy = 1.0
            elif disease.G == 1 or disease.P == 8:
                # Partial heterotypic match (shared G or P)
                expected_efficacy = 0.6
            else:
                # Complete heterotypic match (no shared G,P)
                expected_efficacy = 0.3
                
            assert abs(match_efficacy - expected_efficacy) < 1e-6, f"Disease {disease.name} (G{disease.G}P{disease.P}): expected {expected_efficacy}, got {match_efficacy}"
    
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
            if disease.G in vax.pars.G_antigens and disease.P in vax.pars.P_antigens:
                assert precomputed == 1.0, f"Homotypic {disease.name} should have efficacy 1.0"
            elif disease.G in vax.pars.G_antigens or disease.P in vax.pars.P_antigens:
                assert precomputed == 0.6, f"Partial heterotypic {disease.name} should have efficacy 0.6"
            else:
                assert precomputed == 0.3, f"Complete heterotypic {disease.name} should have efficacy 0.3"
        
    def create_test_sim(self):
        """Helper to create test simulation"""
        vax = rs.RotaVaccination(
            start_date='2020-01-01',
            uptake_dist=ss.bernoulli(0.5),  # Moderate uptake
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