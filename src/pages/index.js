import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

/** Canonical links from [type1compute.com](https://www.type1compute.com/) */
const OFFICIAL_SITE = 'https://www.type1compute.com/';
const OFFICIAL_APPLY_CALL =
  'https://meetings-na2.hubspot.com/param/discovery-call';
const GITHUB_ORG = 'https://github.com/type1compute';
const LINKEDIN_COMPANY =
  'https://www.linkedin.com/company/type-1-compute';
const PUBLISHED_DOCS = 'https://type1compute.github.io/docs';

const SITE_SECTION = {
  /** Open-source signals carousel */
  signals: `${OFFICIAL_SITE}#signals`,
  contact: `${OFFICIAL_SITE}#contact`,
  process: `${OFFICIAL_SITE}#principles`,
};

const pdfOnSite = (filename) => `${OFFICIAL_SITE}pdfs/${filename}.pdf`;

const publishedDocsUrl = (docPath) =>
  `${PUBLISHED_DOCS}${docPath.startsWith('/') ? docPath : `/${docPath}`}`;

const SHIPPED = [
  {
    index: '01',
    title: 'TALON IR ecosystem',
    href: '/talon-ecosystem/introduction',
    githubRepo: GITHUB_ORG,
    body:
      'Neuromorphic intermediate representation for Type 1 Compute hardware. SDK with PyTorch bridge, HDF5 serialization, and visualization—export models without owning every hardware detail.',
    tags: '// SDK · PYTORCH BRIDGE · NEUROMORPHIC IR',
  },
  {
    index: '02',
    title: 'Spike-Space-T1C',
    href: '/spike-space-t1c/intro',
    githubRepo: 'https://github.com/type1compute/SPIKE-SPACE-T1C',
    body:
      'Unsupervised satellite detection from asynchronous event streams. Three-layer spiking convolutional network trained with STDP—built for space domain awareness.',
    tags: '// EVENT-DRIVEN · STDP · INSTANCE SEGMENTATION',
  },
  {
    index: '03',
    title: 'Spectrum Analyzer',
    href: '/spectrum-analyzer/intro',
    githubRepo: 'https://github.com/type1compute/Spectrum-Analyzer',
    body:
      'Detects and classifies 11 radar signal types from spectrograms in real time. ResNet-18 backbone on a modified YOLO-style architecture for RF workloads.',
    tags: '// RADAR · RF CLASSIFICATION · REAL-TIME',
  },
  {
    index: '04',
    title: 'SpikeYoloV8-Tracker',
    href: '/spikeyolo-v8-tracker/introduction',
    githubRepo: 'https://github.com/type1compute/SpikeYoloV8-Tracker',
    body:
      'End-to-end pipeline for real-time object detection and tracking on event camera data. BICLab ECCV 2024 lineage with ByteTracker for temporal continuity.',
    tags: '// OBJECT DETECTION · BYTETRACKER · EVENT CAMERA',
  },
];

const RESEARCH = [
  {
    category: 'Object detection',
    title: 'Event-based object detection',
    highlight: '5.7× efficiency gain',
    blurb:
      'SpikeYOLO: track fast-moving threats in real time using far less power than conventional AI stacks.',
    pdfSlug: 'object-detection',
    thumbFile: 'object-detection.png',
  },
  {
    category: 'Human–machine interface',
    title: 'Gesture recognition',
    highlight: '75.76 GOP/s/W',
    blurb:
      'Energy efficiency measured on neuromorphic-emulated FPGA hardware vs. edge GPUs and CPUs.',
    pdfSlug: 'gesture-recognition',
    thumbFile: 'gesture-recognition.png',
  },
  {
    category: 'Autonomous systems',
    title: 'UAV edge control',
    highlight: '7× over Jetson Nano',
    blurb:
      'Low-SWaP inference for platforms where every watt counts.',
    pdfSlug: 'uav-control',
    thumbFile: 'uav-control.png',
  },
  {
    category: 'Space & radiation',
    title: 'Radiation-tolerant compute',
    highlight: '5× higher MTBF',
    blurb:
      'Deterministic, low-latency inference for harsh environments.',
    pdfSlug: 'radiation-tolerant',
    thumbFile: 'radiation-tolerant.png',
  },
];

const PROCESS = [
  {
    step: '03.1 Intake',
    title: 'Bring your model',
    body:
      'Hand us your PyTorch or ONNX model as-is. No retraining. No pipeline changes. Your training workflow stays the same.',
  },
  {
    step: '03.2 Deployment',
    title: '10× more efficient',
    body:
      'We handle conversion, optimization, and FPGA deployment. Your team works with the outcome—no HDL expertise required on your side.',
  },
  {
    step: '03.3 Ongoing',
    title: 'Managed service',
    body:
      'We maintain the deployment after handoff. Optimization continues as your workload evolves.',
  },
];

const VERTICALS = [
  {
    name: 'Defense',
    status: 'Active',
    desc: 'SWaP-constrained platforms, autonomous systems, EW, radiation-tolerant compute.',
  },
  {
    name: 'Telecom',
    status: 'Accepting partners',
    desc: '5G baseband, RF classification, sparse signal processing on FPGAs.',
  },
  {
    name: 'Industrial',
    status: 'Accepting partners',
    desc: 'Vibration, acoustic, and thermal sensor fusion at the edge.',
  },
  {
    name: 'Medical',
    status: 'Research stage',
    desc: 'EEG, EMG, and implantable-device inference—defining the problem together.',
  },
];

function HomepageHeader() {
  const logoSrc = useBaseUrl('/img/website/t1c-logo.png');
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroBrand}>
          <a
            href={OFFICIAL_SITE}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.heroLogoLink}>
            <img
              src={logoSrc}
              alt="Type 1 Compute"
              className={styles.heroLogo}
              width={200}
              height={48}
              decoding="async"
            />
          </a>
        </div>
        <p className={styles.heroEyebrow}>Your model. Your hardware.</p>
        <Heading as="h1" className={styles.heroTitle}>
          10× more efficient FPGA AI inference
        </Heading>
        <p className={styles.heroSubtitle}>
          We convert your existing AI models to run on FPGAs at a fraction of the
          power—no retraining, no new hardware, no GPU dependency.{' '}
          <Link
            href={SITE_SECTION.process}
            target="_blank"
            rel="noopener noreferrer">
            How we work on type1compute.com →
          </Link>
        </p>
        <p className={styles.heroTrustLine}>
          Currently engaged with tier 1 defense primes and international systems
          integrators.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/intro">
            Explore documentation →
          </Link>
          <button
            type="button"
            className="button button--outline button--secondary button--lg"
            onClick={() =>
              document
                .getElementById('performance')
                ?.scrollIntoView({behavior: 'smooth', block: 'start'})
            }>
            View performance →
          </button>
          <Link
            className="button button--outline button--secondary button--lg"
            href={OFFICIAL_SITE}
            target="_blank"
            rel="noopener noreferrer">
            type1compute.com →
          </Link>
        </div>
      </div>
    </header>
  );
}

function ResearchCard({item}) {
  const thumbSrc = useBaseUrl(`/img/research-briefs/${item.thumbFile}`);
  const pdfHref = pdfOnSite(item.pdfSlug);
  return (
    <article className={styles.researchCard}>
      <a
        href={pdfHref}
        target="_blank"
        rel="noopener noreferrer"
        className={styles.researchThumbLink}>
        <img
          src={thumbSrc}
          alt={`First page preview: ${item.title} (PDF brief on type1compute.com)`}
          className={styles.researchThumb}
          loading="lazy"
          decoding="async"
        />
      </a>
      <div className={styles.researchCardBody}>
        <div className={styles.researchCategory}>{item.category}</div>
        <div className={styles.researchTitle}>{item.title}</div>
        <div className={styles.researchHighlight}>{item.highlight}</div>
        <p className={styles.researchBlurb}>{item.blurb}</p>
        <Link
          href={pdfHref}
          className={styles.researchPdfCta}
          target="_blank"
          rel="noopener noreferrer">
          Open PDF brief →
        </Link>
      </div>
    </article>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Type 1 Compute converts your PyTorch or ONNX models for efficient FPGA inference—10× more efficient than typical edge GPUs. Open docs for TALON IR, Spike-Space-T1C, Spectrum Analyzer, and SpikeYoloV8-Tracker.">
      <HomepageHeader />
      <main>
        <section className={styles.section} aria-labelledby="shipped-heading">
          <div className="container">
            <p className={styles.sectionLabel}>01 / Open source</p>
            <Heading as="h2" id="shipped-heading" className={styles.sectionTitle}>
              What we&apos;ve shipped
            </Heading>
            <p className={clsx('text--center', styles.sectionIntro)}>
              Documentation in this site matches the open-source lineup featured on{' '}
              <Link href={SITE_SECTION.signals} target="_blank" rel="noopener noreferrer">
                type1compute.com
              </Link>
              . Each project is also listed on our{' '}
              <Link href={GITHUB_ORG} target="_blank" rel="noopener noreferrer">
                GitHub organization
              </Link>
              , with a mirrored build at{' '}
              <Link href={PUBLISHED_DOCS} target="_blank" rel="noopener noreferrer">
                GitHub Pages docs
              </Link>
              .
            </p>
            <div className={styles.shippedGrid}>
              {SHIPPED.map((item) => (
                <article key={item.index} className={styles.shippedCard}>
                  <div className={styles.shippedIndex}>No. {item.index}</div>
                  <Heading as="h3" className={styles.shippedCardTitle}>
                    <Link to={item.href}>{item.title}</Link>
                  </Heading>
                  <p className={styles.shippedCardBody}>{item.body}</p>
                  <div className={styles.shippedTags}>{item.tags}</div>
                  <div className={styles.shippedCardLink}>
                    <Link to={item.href}>Read the docs →</Link>
                  </div>
                  <div className={styles.shippedMetaLinks}>
                    <Link
                      href={item.githubRepo}
                      target="_blank"
                      rel="noopener noreferrer">
                      Repository
                    </Link>
                    {' · '}
                    <Link
                      href={publishedDocsUrl(item.href)}
                      target="_blank"
                      rel="noopener noreferrer">
                      GitHub Pages
                    </Link>
                    {' · '}
                    <Link
                      href={SITE_SECTION.signals}
                      target="_blank"
                      rel="noopener noreferrer">
                      Company site
                    </Link>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section
          id="performance"
          className={styles.performanceSection}
          aria-labelledby="performance-heading">
          <div className="container">
            <p className={styles.sectionLabel}>Benchmark</p>
            <Heading as="h2" id="performance-heading" className="text--center margin-bottom--lg">
              Performance comparison
            </Heading>
            <p className="text--center margin-bottom--xl">
              Gesture recognition benchmark (DVS128 dataset)
              <br />
              Energy efficiency on neuromorphic-emulated FPGA hardware.{' '}
              <Link
                href={pdfOnSite('gesture-recognition')}
                target="_blank"
                rel="noopener noreferrer">
                Gesture recognition PDF brief →
              </Link>
            </p>
            <div className={styles.performanceChart}>
              <div className={styles.performanceBar}>
                <div className={styles.performanceLabel}>Type 1 Compute FPGA</div>
                <div className={styles.performanceBarRow}>
                  <div className={styles.performanceBarContainer}>
                    <div className={styles.performanceBarFill} style={{width: '100%'}} />
                  </div>
                  <span className={styles.performanceValue}>75.76 GOP/s/W</span>
                </div>
              </div>
              <div className={styles.performanceBar}>
                <div className={styles.performanceLabel}>Jetson Nano</div>
                <div className={styles.performanceBarRow}>
                  <div className={styles.performanceBarContainer}>
                    <div className={styles.performanceBarFill} style={{width: '10.6%'}} />
                  </div>
                  <span className={styles.performanceValue}>8.00 GOP/s/W</span>
                </div>
              </div>
              <div className={styles.performanceBar}>
                <div className={styles.performanceLabel}>RTX 3060</div>
                <div className={styles.performanceBarRow}>
                  <div className={styles.performanceBarContainer}>
                    <div className={styles.performanceBarFill} style={{width: '5.0%'}} />
                  </div>
                  <span className={styles.performanceValue}>3.81 GOP/s/W</span>
                </div>
              </div>
              <div className={styles.performanceBar}>
                <div className={styles.performanceLabel}>Intel i9</div>
                <div className={styles.performanceBarRow}>
                  <div className={styles.performanceBarContainer}>
                    <div className={styles.performanceBarFill} style={{width: '0.4%'}} />
                  </div>
                  <span className={styles.performanceValue}>0.31 GOP/s/W</span>
                </div>
              </div>
            </div>
            <p className="text--center margin-top--lg">
              <strong>
                244× more efficient than CPU, 9.5× more efficient than Jetson
              </strong>
            </p>
          </div>
        </section>

        <section className={styles.section} aria-labelledby="research-heading">
          <div className="container">
            <p className={styles.sectionLabel}>02 / Research</p>
            <Heading as="h2" id="research-heading" className="text--center margin-bottom--lg">
              Published work
            </Heading>
            <p className="text--center margin-bottom--xl">
              Technical results across sparse inference, event-driven perception, and
              autonomous edge compute—additional medical and industrial applications
              available on request.
            </p>
            <div className={styles.researchGrid}>
              {RESEARCH.map((r) => (
                <ResearchCard key={r.title} item={r} />
              ))}
            </div>
          </div>
        </section>

        <section className={styles.sectionMuted} aria-labelledby="process-heading">
          <div className="container">
            <p className={styles.sectionLabel}>03 / Process</p>
            <Heading as="h2" id="process-heading" className="text--center margin-bottom--lg">
              How we work
            </Heading>
            <p className="text--center margin-bottom--lg">
              <Link href={SITE_SECTION.process} target="_blank" rel="noopener noreferrer">
                Full process narrative on type1compute.com →
              </Link>
            </p>
            <div className={styles.processGrid}>
              {PROCESS.map((p) => (
                <div key={p.step} className={styles.processStep}>
                  <div className={styles.processStepIndex}>{p.step}</div>
                  <Heading as="h3" className={styles.processStepTitle}>
                    {p.title}
                  </Heading>
                  <p className={styles.processStepBody}>{p.body}</p>
                </div>
              ))}
            </div>
            <p className="text--center margin-top--lg" style={{fontSize: '0.9rem'}}>
              On the roadmap: custom ASIC, 100× efficiency over Jetson, priority access
              for pipeline partners.
            </p>
          </div>
        </section>

        <section className={styles.section} aria-labelledby="work-heading">
          <div className="container">
            <p className={styles.sectionLabel}>04 / Contact</p>
            <Heading as="h2" id="work-heading" className="text--center margin-bottom--lg">
              Work with us
            </Heading>
            <p className="text--center margin-bottom--lg">
              <Link href={SITE_SECTION.contact} target="_blank" rel="noopener noreferrer">
                Contact, backers, and apply on type1compute.com →
              </Link>
            </p>
            <div className={styles.verticalsGrid}>
              {VERTICALS.map((v) => (
                <div key={v.name} className={styles.verticalCard}>
                  <div className={styles.verticalName}>{v.name}</div>
                  <div className={styles.verticalStatus}>{v.status}</div>
                  <p className={styles.verticalDesc}>{v.desc}</p>
                </div>
              ))}
            </div>
            <div className={styles.contactStrip}>
              <p>
                <a href="mailto:support@type1compute.com">support@type1compute.com</a>
                {' · '}
                We respond within 48 hours.
              </p>
              <div className={styles.contactActions}>
                <Link
                  className="button button--primary button--lg"
                  href={OFFICIAL_APPLY_CALL}
                  target="_blank"
                  rel="noopener noreferrer">
                  Apply — book a 30-minute call →
                </Link>
                <Link
                  className="button button--outline button--primary button--lg"
                  href={OFFICIAL_SITE}
                  target="_blank"
                  rel="noopener noreferrer">
                  Company site →
                </Link>
              </div>
              <div className={styles.socialRow}>
                <Link href={GITHUB_ORG} target="_blank" rel="noopener noreferrer">
                  GitHub
                </Link>
                {' · '}
                <Link href={LINKEDIN_COMPANY} target="_blank" rel="noopener noreferrer">
                  LinkedIn
                </Link>
              </div>
            </div>
          </div>
        </section>

        <p className={styles.footerTagline}>
          Edge AI inference for platforms where every watt counts.
        </p>
      </main>
    </Layout>
  );
}
