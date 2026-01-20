# System Architecture Diagrams

## Overall Pipeline

```mermaid
flowchart TB
    subgraph Stage1["Stage 1: Pre-training (Large-scale Data)"]
        MRI_DATA[(MRI Data<br/>BraTS 1,242)]
        GENE_DATA[(Gene Data<br/>CGGA ~1,000)]

        MRI_DATA --> M1[M1: SwinUNETR<br/>Encoder]
        GENE_DATA --> MG[MG: VAE<br/>Encoder]

        M1 --> MRI_FEAT[768-dim<br/>MRI Features]
        MG --> GENE_FEAT[64-dim<br/>Gene Features]
    end

    subgraph Stage2["Stage 2: Multimodal Fusion (72 patients)"]
        PROTEIN_DATA[(Protein Data<br/>TCGA 229-dim)]

        MRI_FEAT --> |Transfer| MM[MM: Cross-Modal<br/>Attention]
        GENE_FEAT --> |Transfer| MM
        PROTEIN_DATA --> MM

        MM --> PRED[Predictions<br/>Survival, IDH, MGMT...]
    end

    style M1 fill:#0099CC
    style MG fill:#00AA66
    style MM fill:#9933FF
```

## M1 Model: MRI Encoder

```mermaid
flowchart LR
    subgraph Input
        T1[T1]
        T1CE[T1ce]
        T2[T2]
        FLAIR[FLAIR]
    end

    subgraph SwinUNETR
        ENC[Swin Transformer<br/>Encoder]
        DEC[UNet<br/>Decoder]
    end

    subgraph Output
        SEG[Segmentation<br/>4 classes]
        FEAT[Features<br/>768-dim]
    end

    T1 & T1CE & T2 & FLAIR --> ENC
    ENC --> DEC --> SEG
    ENC --> FEAT

    style ENC fill:#0099CC
    style FEAT fill:#0099CC
```

## MG Model: Gene VAE Encoder

```mermaid
flowchart LR
    subgraph Input
        GENES[500 Genes]
    end

    subgraph VAE
        ENC[Encoder]
        MU[μ]
        SIGMA[σ²]
        Z[z = μ + σ×ε<br/>64-dim latent]
        DEC[Decoder]
    end

    subgraph Output
        RECON[Reconstruction]
        TASKS[Task Heads<br/>Survival, Grade...]
        TRANSFER[Transfer to MM]
    end

    GENES --> ENC
    ENC --> MU & SIGMA
    MU & SIGMA --> Z
    Z --> DEC --> RECON
    Z --> TASKS
    Z --> TRANSFER

    style Z fill:#00AA66
```

## MM Model: Cross-Modal Attention

```mermaid
flowchart TB
    subgraph Inputs
        MRI[MRI Features<br/>768-dim]
        GENE[Gene Features<br/>64-dim]
        PROT[Protein<br/>229-dim]
    end

    subgraph Projection
        MRI_P[Project to 256]
        GENE_P[Project to 256]
        PROT_P[Project to 256]
    end

    subgraph Attention["Cross-Modal Attention (8 heads)"]
        ATT[Multi-Head<br/>Self-Attention]
    end

    subgraph Outputs
        SURV[Survival<br/>C-Index 0.61]
        IDH[IDH<br/>AUC 0.878]
        MGMT[MGMT]
        GRADE[Grade]
    end

    MRI --> MRI_P
    GENE --> GENE_P
    PROT --> PROT_P

    MRI_P & GENE_P & PROT_P --> ATT
    ATT --> SURV & IDH & MGMT & GRADE

    style ATT fill:#9933FF
```

## Transfer Learning Strategy

```mermaid
flowchart LR
    subgraph PreTrain["Pre-training Phase"]
        BraTS[BraTS<br/>1,242 patients] --> M1_PT[M1 Training]
        CGGA[CGGA<br/>~1,000 patients] --> MG_PT[MG Training]
    end

    subgraph Transfer["Transfer Phase"]
        M1_PT --> |Freeze Encoder| MM_TRAIN[MM Training]
        MG_PT --> |Freeze Encoder| MM_TRAIN
        TCGA[TCGA<br/>72 patients] --> MM_TRAIN
    end

    subgraph Result
        MM_TRAIN --> FINAL[Multimodal<br/>Predictions]
    end

    style M1_PT fill:#0099CC
    style MG_PT fill:#00AA66
    style MM_TRAIN fill:#9933FF
```

## Separation Strategy (Training vs Inference)

```mermaid
flowchart TB
    subgraph Training["Training Pipeline"]
        GENE_T[Gene Expression] --> VAE_T[VAE Encoder]
        VAE_T --> LATENT_T[64-dim Latent]
        LATENT_T --> PRED_T[Predictions]
        LATENT_T --> |Transfer| MM_T[MM Model]

        PATH_X[Pathway ❌<br/>Not Used]
    end

    subgraph Inference["Inference Pipeline"]
        GENE_I[Gene Expression] --> VAE_I[VAE Encoder<br/>Frozen]
        GENE_I --> SSGSEA[ssGSEA]

        VAE_I --> PRED_I[Predictions]
        SSGSEA --> PATH_I[50 Hallmark<br/>Pathways]
        PATH_I --> INTERP[Interpretation<br/>for Clinicians]
    end

    style VAE_T fill:#00AA66
    style VAE_I fill:#00AA66
    style PATH_X fill:#FF6666
    style INTERP fill:#FFB366
```
