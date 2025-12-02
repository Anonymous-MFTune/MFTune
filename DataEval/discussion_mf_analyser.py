from utils.multi_fidelity_analyser import MultiFidelityAnalyser

def main():

    """Behavior analysis for GA with Multi-fideity"""

    system = 'mysql'

    for run in range(0, 1):
        multi_fidelity_analyser = MultiFidelityAnalyser(run, system)
        multi_fidelity_analyser.analyze_filtering_stages()



        multi_fidelity_analyser.best_ind_convergence_visualization()
        multi_fidelity_analyser.average_pop_convergence_visualization()
        # multi_fidelity_analyser.evo_landscape_visualization(run)

        multi_fidelity_analyser.hf_best_ind_convergence_visualization()
        multi_fidelity_analyser.hf_average_pop_convergence_visualization()


if __name__ == "__main__":
    main()
