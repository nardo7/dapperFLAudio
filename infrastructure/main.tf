terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.8.0"
    }
  }
}

locals {
  project_name = "ddl-project"
  project_id   = "ddl-project-456412"
  region       = "us-west4"
  zone         = "us-west4-b"
}

provider "google" {
  project = local.project_id
  region  = local.region
  zone    = local.zone
}



# This code is compatible with Terraform 4.25.0 and versions that are backwards compatible to 4.25.0.
# For information about validating this Terraform code, see https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/google-cloud-platform-build#format-and-validate-the-configuration

resource "google_compute_instance" "unibern-ddl-project" {

  machine_type = "n1-standard-4"
  name         = "unibern-${local.project_name}"
  tags         = ["http-server", "https-server"]
  zone         = local.zone

  depends_on = [google_compute_disk.unibern-ddl-project]

  boot_disk {
    auto_delete = true
    device_name = "unibern-${local.project_name}"
    source      = google_compute_disk.unibern-ddl-project.id


    # initialize_params {
    #   image = "projects/ml-images/global/images/c0-deeplearning-common-cu123-v20240922-debian-11"
    # #   size  = 150
    #   type  = "pd-balanced"
    # }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  guest_accelerator {
    count = 1
    type  = "projects/${local.project_id}/zones/${local.zone}/acceleratorTypes/nvidia-tesla-t4"
  }

  labels = {
    goog-ec-src = "vm_add-tf"
  }


  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    queue_count = 0
    stack_type  = "IPV4_ONLY"
    subnetwork  = "projects/${local.project_id}/regions/${local.region}/subnetworks/default"
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "TERMINATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  service_account {
    email  = "1023006117320-compute@developer.gserviceaccount.com"
    scopes = ["https://www.googleapis.com/auth/devstorage.read_only", "https://www.googleapis.com/auth/logging.write", "https://www.googleapis.com/auth/monitoring.write", "https://www.googleapis.com/auth/service.management.readonly", "https://www.googleapis.com/auth/servicecontrol", "https://www.googleapis.com/auth/trace.append"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

}

resource "google_compute_disk" "unibern-ddl-project" {
  project = local.project_id
  name    = "unibern-${local.project_name}"
  type    = "pd-balanced"
  zone    = local.zone
  size    = 150
  image   = "projects/ml-images/global/images/c0-deeplearning-common-cu123-v20240922-debian-11"
  #   mode = "READ_WRITE"
}
