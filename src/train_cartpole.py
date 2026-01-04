from cartpole_trainer import CartPoleTrainer


def main():
    trainer = CartPoleTrainer(episodes=1000, target_update_frequency=10)
    
    print("Starting CartPole training...")
    scores, rewards = trainer.train()
    
    print("\nTraining completed!")
    print(f"Final average score (last 100 episodes): {sum(scores[-100:]) / 100:.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")
    
    trainer.plot_results()
    trainer.save_model()


if __name__ == "__main__":
    main()

